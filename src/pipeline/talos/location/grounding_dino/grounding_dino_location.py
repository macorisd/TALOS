from typing import Dict, List
import torch
import json

import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="transformers.models.grounding_dino.processing_grounding_dino", 
    lineno=100
)

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from talos.location.base_location import BaseLocator

class GroundingDinoLocator(BaseLocator):
    """
    [Location -> Grounding DINO]
    
    Location stage implementation that leverages the Grounding DINO model.
    """

    STR_PREFIX = "\n[LOCATION | GROUNDING DINO]" # Prefix for logging
    ALIAS = "gdino" # Alias for filenames

    def __init__(
        self,
        grounding_dino_model_name: str = "IDEA-Research/grounding-dino-base",
        score_threshold: float = 0.2
    ):
        """
        Initialize the Grounding DINO locator.
        """
        print(f"{self.STR_PREFIX} Initializing Grounding DINO locator...", end=" ", flush=True)

        # Initialize base class
        super().__init__(score_threshold)

        # Load the processor and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(grounding_dino_model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_name).to(device)

        print("Done.")

    
    # Override
    def json_to_model_prompt(self, tags: List[str]) -> str:
        """
        Converts the tags list to a Grounding DINO prompt.
        """
        # Build the Grounding Dino prompt
        prompt = ". ".join(tags) + "."

        return prompt
    

    # Override
    def model_results_to_json(self, results: Dict) -> List[Dict]:
        """
        Converts the Grounding DINO results to a JSON dict list.
        """
        scores = results.get("scores", torch.tensor([])).tolist()
        boxes = results.get("boxes", torch.tensor([])).tolist()
        labels = results.get("labels", [])
                
        result = []
        for label, bbox, score in zip(labels, boxes, scores):
            obj = {
                "label": label,
                "score": float(score),
                "bbox": {
                    "x_min": bbox[0],
                    "y_min": bbox[1],
                    "x_max": bbox[2],
                    "y_max": bbox[3]
                }
            }
            result.append(obj)
        
        return result
    

    # Override
    def execute(self) -> List[Dict]:
        """
        Execute the Grounding DINO Location.
        """
        print(f"{self.STR_PREFIX} Running Location with Grounding DINO...", end=" ", flush=True)

        # Convert the tags JSON text to a Grounding Dino prompt
        text = self.json_to_model_prompt(self.input_tags)

        # Process and predict
        inputs = self.processor(images=self.input_image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.input_image.size[::-1]]
        )[0]

        print(f"Object detection results:\n\n{results}")

        # Convert the results to a JSON dict list
        results_json = self.model_results_to_json(results)

        # Filter the results based on the confidence threshold
        results_json = self.filter_confidence(results_json, threshold=self.score_threshold)

        # Filter the results based on bounding box properties
        results_json = self.filter_bbox(results_json, self.input_image.width, self.input_image.height, verbose=True)

        # Filter the results based on label coincidence with the tagging stage
        results_json = self.filter_labels(results_json, self.input_tags)

        print(f"{self.STR_PREFIX} JSON results:\n\n{json.dumps(results_json, indent=4)}")

        return results_json
    

def main():
    """
    Main function to run the Grounding DINO locator.
    """
    locator = GroundingDinoLocator()
    locator.load_inputs(input_image_name="avocado.jpeg")

    tags = locator.execute()

    locator.save_outputs(tags)


if __name__ == "__main__":
    main()