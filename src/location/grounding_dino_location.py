import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="transformers.models.grounding_dino.processing_grounding_dino", 
    lineno=100
)

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import time
import json

class GroundingDinoLocator:
    """
    A class to locate objects in an image using the Grounding Dino model.
    """

    STR_PREFIX = "[LOCATION | GDINO]"

    def __init__(
            self,
            grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base",            
            score_threshold: float = 0.2,
            save_file_json: bool = True,
            save_file_jpg: bool = True
    ):
        """
        TODO
        """

        print(f"\n{self.STR_PREFIX} Initializing Grounding DINO object locator...", end=" ")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.score_threshold = score_threshold if score_threshold > 0 else 0.2
        self.save_file_json = save_file_json
        self.save_file_jpg = save_file_jpg

        # Load the processor and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)

        if save_file_json or save_file_jpg:
            # Output location directory path
            self.output_location_dir = os.path.join(
                self.script_dir, 
                "output_location"
            )

            # Create the output directory if it does not exist
            os.makedirs(self.output_location_dir, exist_ok=True)
        
        print("Done.\n")

    def load_image(self, input_image_name: str) -> None:
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ")

        # Input image path
        input_image_path = os.path.join(
            self.script_dir, 
            "..",                
            "input_images",
            input_image_name
        )

        # Load input image
        if os.path.isfile(input_image_path):
            self.input_image = Image.open(input_image_path)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.\n")
        
        print("Done.\n")
    
    def load_tags(self, pipeline_tags: dict = None) -> None:
        print(f"{self.STR_PREFIX} Loading input tags...", end=" ")

        # If pipeline_tags is provided, use it
        if pipeline_tags is not None:
            self.input_tags = pipeline_tags
        
        # Otherwise, read the most recent .json file from output_tags
        else:
            # Input tags directory path
            input_tags_dir = os.path.join(
                self.script_dir, 
                "..",
                "tagging",
                "output_tags"
            )

            # Gather all .json files in input_tags_dir
            json_files = [
                os.path.join(input_tags_dir, f)
                for f in os.listdir(input_tags_dir)
                if f.endswith(".json")
            ]
            if not json_files:
                raise FileNotFoundError(f"{self.STR_PREFIX} No .json files found in {input_tags_dir}\n")

            # Select the most recently modified .json file
            latest_json_path = max(json_files, key=os.path.getmtime)
            
            # Extract the filename
            filename = os.path.basename(latest_json_path)
            print("Most recent .json file:", filename, end="... ")
            
            # Read the content of the file
            with open(latest_json_path, "r", encoding="utf-8") as f:
                self.input_tags = json.load(f)

        print("Done.\n")        
    
    def json_to_gdino_prompt(self, tags: dict) -> str:
        """
        Converts the tags dictionary to a Grounding DINO prompt.
        """
        # Extract the tags from the dictionary
        tags_list = [tag for tag in tags.values()]
        
        # Build the Grounding Dino prompt
        prompt = ". ".join(tags_list) + "."

        return prompt
    
    def gdino_results_to_json(self, results: dict) -> dict:
        """
        Converts the Grounding DINO results to a JSON dict.
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
    
    def filter_confidence(self, results: dict, threshold: float) -> dict:
        """
        Filters the results based on the confidence threshold.
        """
        filtered_results = [result for result in results if result["score"] > threshold]
        return filtered_results        

    def filter_bbox(self, results_json: dict, image_width, image_height, padding: float = 30, ratio: float = 0.9 , verbose: bool = False):
        def is_similar_bbox(bbox1, bbox2, padding):
            return (abs(bbox1['x_min'] - bbox2['x_min']) <= padding and
                (abs(bbox1['y_min'] - bbox2['y_min']) <= padding) and
                (abs(bbox1['x_max'] - bbox2['x_max']) <= padding) and
                (abs(bbox1['y_max'] - bbox2['y_max']) <= padding))

        def is_bbox_contained(bbox1, bbox2, padding):        
            return (bbox1['x_min'] >= bbox2['x_min'] - padding and
                    bbox1['y_min'] >= bbox2['y_min'] - padding and
                    bbox1['x_max'] <= bbox2['x_max'] + padding and
                    bbox1['y_max'] <= bbox2['y_max'] + padding)

        # 1. Discard practically equal bounding boxes with lower score
        i = 0
        while i < len(results_json):
            j = i + 1
            while j < len(results_json):
                if is_similar_bbox(results_json[i]['bbox'], results_json[j]['bbox'], padding):
                    if results_json[i]['score'] > results_json[j]['score']:
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[j]['label']} with score {results_json[j]['score']} [1]")
                        results_json.pop(j)
                    else:
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[i]['label']} with score {results_json[i]['score']} [1]")
                        results_json.pop(i)
                        i -= 1
                        break
                else:
                    j += 1
            i += 1

        # 2. Discard bounding boxes that are too large
        filtered_results = []
        for result in results_json:
            width = result['bbox']['x_max'] - result['bbox']['x_min']
            height = result['bbox']['y_max'] - result['bbox']['y_min']

            if not (width >= image_width * ratio and height >= image_height * ratio):
                filtered_results.append(result)
            elif verbose:
                print(f"{self.STR_PREFIX} Discarded: {result['label']} with score {result['score']} [2]")
        results_json = filtered_results

        # 3. Discard bounding boxes with the same label and one fully contained in the other (discards the bigger one)
        i = 0
        while i < len(results_json):
            j = i + 1
            while j < len(results_json):
                if results_json[i]['label'] == results_json[j]['label']:
                    if is_bbox_contained(results_json[i]['bbox'], results_json[j]['bbox'], padding):
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[j]['label']} with score {results_json[j]['score']} [3]")
                        results_json.pop(j)
                        continue
                    elif is_bbox_contained(results_json[j]['bbox'], results_json[i]['bbox'], padding):
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[i]['label']} with score {results_json[i]['score']} [3]")
                        results_json.pop(i)
                        i -= 1
                        break
                j += 1
            i += 1

        return results_json
    
    def draw_bounding_boxes(self, results: dict, padding: int = None) -> Image:
        """
        Draws bounding boxes around the detected objects in the image.
        """    
        image = self.input_image.copy()
        draw = ImageDraw.Draw(image)        
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # Draw bounding boxes for each detected object
        for obj in results:
            label = obj.get("label", "unknown")
            score = obj.get("score", 0.0)
            bbox = obj.get("bbox", {})
            
            # Extract bounding box coordinates
            x_min = int(bbox.get("x_min", 0))
            y_min = int(bbox.get("y_min", 0))
            x_max = int(bbox.get("x_max", 0))
            y_max = int(bbox.get("y_max", 0))
            
            # Draw the bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
            # Draw the label and score
            text = f"{label}: {score:.2f}"
            draw.text((x_min, y_min), text, fill="red", font=font)

        # If padding is not None, draw a green rectangle that represents the padding for reference
        if padding is not None:
            _, image_height = image.size

            rect_width = padding
            rect_height = 30
            
            # Define the coordinates for the rectangle
            rect_x_min = 0
            rect_y_min = image_height - rect_height
            rect_x_max = rect_width
            rect_y_max = image_height
            
            # Draw the green rectangle
            draw.rectangle([rect_x_min, rect_y_min, rect_x_max, rect_y_max], outline="green", fill="green", width=2)
            
            # Draw the "padding" text
            text = "padding"
            text_x = rect_x_min + 5
            text_y = rect_y_min - 30
            draw.text((text_x, text_y), text, fill="green", font=font)

        return image
    
    def run(self) -> dict:
        """
        TODO
        """
        print(f"{self.STR_PREFIX} Running Grounding DINO object bounding box locator...", end=" ")

        # Convert the tags JSON text to a Grounding Dino prompt
        text = self.json_to_gdino_prompt(self.input_tags)

        # Process and predict
        inputs = self.processor(images=self.input_image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.input_image.size[::-1]]
        )[0]

        print(f"Object detection results:\n\n{results}\n")

        # Convert the results to JSON format
        results_json = self.gdino_results_to_json(results)

        # Filter the results based on the confidence threshold
        results_json = self.filter_confidence(results_json, threshold=self.score_threshold)

        # Filter the results based on bounding box properties
        results_json = self.filter_bbox(results_json, self.input_image.width, self.input_image.height, verbose=True)

        print(f"{self.STR_PREFIX} JSON results:\n\n{json.dumps(results_json, indent=4)}\n")

        # Save the results to a JSON file and/or an image file
        if self.save_file_json or self.save_file_jpg:
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            if self.save_file_json:
                # Prepare JSON output file
                output_filename_json = f"location_gdino_{timestamp}.json"
                output_file_json = os.path.join(self.output_location_dir, output_filename_json)

                # Save the results to a JSON file
                with open(output_file_json, "w", encoding="utf-8") as f:
                    json.dump(results_json, f, indent=4)
                    print(f"{self.STR_PREFIX} Object bounding box location JSON results saved to: {output_file_json}\n")

            if self.save_file_jpg:
                # Prepare JPG output file
                output_filename_jpg = f"location_gdino_{timestamp}.jpg"
                output_file_jpg = os.path.join(self.output_location_dir, output_filename_jpg)

                # Draw bounding boxes around the detected objects
                results_image = self.draw_bounding_boxes(results=results_json, padding=30)

                # Save the image with bounding boxes
                results_image.save(output_file_jpg)
                print(f"{self.STR_PREFIX} Bounding box location image saved to: {output_file_jpg}")

        return results_json

def main():
    """
    Main function for the Grounding DINO Locator.
    """
    locator = GroundingDinoLocator(        
        score_threshold=0
    )
    locator.load_image("desk.jpg")
    locator.load_tags()
    locator.run()


if __name__ == "__main__":
    main()