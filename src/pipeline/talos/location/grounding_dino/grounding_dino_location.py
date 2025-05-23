from typing import Dict, List
import torch

import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="transformers.models.grounding_dino.processing_grounding_dino", 
    lineno=100
)

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from pipeline.talos.location.base_location import BaseLocator

class GroundingDinoLocator(BaseLocator):
    """
    [Location -> Grounding DINO]
    
    Location stage implementation that leverages the Grounding DINO model.
    """

    STR_PREFIX = "\n[LOCATION | GROUNDING DINO]" # Prefix for logging
    ALIAS = "gdino" # Alias for filenames

    def __init__(
        self,
        grounding_dino_model_name: str = "IDEA-Research/grounding-dino-base"
    ):
        """
        Initialize the Grounding DINO locator.
        """
        print(f"{self.STR_PREFIX} Initializing Grounding DINO locator...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Load the processor and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(grounding_dino_model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_name).to(device)

        print("Done.")

    # Override from BaseLocator
    def json_to_model_prompt(self, tags: List[str]) -> str:
        """
        Converts the tags list to a Grounding DINO prompt.
        """
        # Build the Grounding Dino prompt
        prompt = ". ".join(tags) + "."

        return prompt
    
    # Override from BaseLocator
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
    
    # Override from ILocationStrategy -> BaseLocator
    def execute(self) -> List[Dict]:
        """
        Execute the Grounding DINO Location.
        """
        print(f"{self.STR_PREFIX} Running Location with Grounding DINO...", end=" ", flush=True)

        # Parent execution method
        results = self.execute_location()

        return results
    
    # Override from BaseLocator
    def locate_bboxes(self, text: str) -> Dict:
        """
        Use the Grounding DINO model to locate bounding boxes in the image.
        
        This method will be called by the superclass.
        """
        # Process and predict
        inputs = self.processor(images=self.input_image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.input_image.size[::-1]]
        )[0]

        # Clear the GPU cache
        torch.cuda.empty_cache()

        print(f"Object detection results:\n\n{results}")

        return results


def main():
    """
    Main function to run the Location with Grounding DINO.
    """
    locator = GroundingDinoLocator()
    locator.load_inputs(input_image_name="avocado.jpeg")

    tags = locator.execute()

    locator.save_outputs(tags)


if __name__ == "__main__":
    main()