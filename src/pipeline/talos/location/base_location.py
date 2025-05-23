from abc import abstractmethod
import json
import os
import time
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from pipeline.strategy.strategy import ILocationStrategy
from pipeline.config.config import (
    config,
    SAVE_FILES,
    LOCATION_SCORE_THRESHOLD,
    LOCATION_PADDING_RATIO,
    LOCATION_LARGE_BBOX_RATIO
)
from pipeline.config.paths import (
    INPUT_IMAGES_DIR,
    OUTPUT_TAGS_DIR,
    OUTPUT_LOCATION_DIR
)


class BaseLocator(ILocationStrategy):
    """
    [Location]
    
    Base class for Location implementations.
    """

    def __init__(self):
        """
        Initialize the base locator.
        """
        if config.get(SAVE_FILES):
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_LOCATION_DIR, exist_ok=True)

    # Override from ILocationStrategy
    def load_inputs(
        self,
        input_image_name: str = None,
        input_image: np.ndarray = None,
        input_tags: List[str] = None
    ) -> None:
        """
        Load the Location inputs.
        """
        if input_image_name is None and input_image is None:
            raise ValueError(f"{self.STR_PREFIX} Either input_image_name or input_image must be provided.")
        
        if input_image is not None:
            self.set_image(input_image)
        else:
            self.load_image(input_image_name)

        # Load the input tags
        self.load_tags(input_tags)

    # Override from ILocationStrategy
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image for the Location stage.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ", flush=True)

        # Input image path
        input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        # Load input image
        if os.path.isfile(input_image_path):
            self.input_image = Image.open(input_image_path)
            
            if self.input_image.mode != "RGB":
                self.input_image = self.input_image.convert("RGB")
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.")
        
        print("Done.")

    # Override from ILocationStrategy
    def set_image(self, input_image: np.ndarray) -> None:
        """
        Set the input image.
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.input_image = Image.fromarray(rgb_image)

    # Override from ILocationStrategy
    def load_tags(self, input_tags: List[str] = None) -> None:
        """
        Load the input tags (output from the Tagging stage) for the Location stage.
        """
        print(f"{self.STR_PREFIX} Loading input tags...", end=" ", flush=True)

        # If input_tags is provided, use it
        if input_tags is not None:
            self.input_tags = input_tags
        
        # Otherwise, read the most recent .json file from output_tags
        else:
            # Gather all .json files in the output tags directory
            json_files = [
                os.path.join(OUTPUT_TAGS_DIR, f)
                for f in os.listdir(OUTPUT_TAGS_DIR)
                if f.endswith(".json")
            ]
            if not json_files:
                raise FileNotFoundError(f"{self.STR_PREFIX} No .json files found in {input_tags_dir}")

            # Select the most recently modified .json file
            latest_json_path = max(json_files, key=os.path.getmtime)
            
            # Extract the filename
            filename = os.path.basename(latest_json_path)
            print("Most recent .json file: ", filename, end="... ")
            
            # Read the content of the file
            with open(latest_json_path, "r", encoding="utf-8") as f:
                self.input_tags = json.load(f)

        print("Done.")

    @abstractmethod # from ILocationStrategy
    def execute(self):
        raise NotImplementedError("execute method must be implemented in subclasses.")

    def __filter_confidence(self, results: Dict) -> Dict:
        """
        Filters the results based on the confidence threshold.
        """
        filtered_results = [result for result in results if result["score"] > config.get(LOCATION_SCORE_THRESHOLD)]
        return filtered_results        

    def __filter_bbox(
            self, 
            results_json: dict,
            image_width,
            image_height,
            verbose: bool = True
    ):
        """
        Filters the results based on bounding box properties.
        """
        padding = image_width * config.get(LOCATION_PADDING_RATIO)
        
        def __is_similar_bbox(bbox1, bbox2, padding):
            """
            Check if two bounding boxes are similar based on a padding.
            """
            return (abs(bbox1['x_min'] - bbox2['x_min']) <= padding and
                (abs(bbox1['y_min'] - bbox2['y_min']) <= padding) and
                (abs(bbox1['x_max'] - bbox2['x_max']) <= padding) and
                (abs(bbox1['y_max'] - bbox2['y_max']) <= padding))

        def __is_bbox_contained(bbox1, bbox2, padding):
            """
            Check if one bounding box is contained within another with a padding.
            """
            return (bbox1['x_min'] >= bbox2['x_min'] - padding and
                    bbox1['y_min'] >= bbox2['y_min'] - padding and
                    bbox1['x_max'] <= bbox2['x_max'] + padding and
                    bbox1['y_max'] <= bbox2['y_max'] + padding)

        # 1. Discard practically equal bounding boxes with lower score
        i = 0
        while i < len(results_json):
            j = i + 1
            while j < len(results_json):
                if __is_similar_bbox(results_json[i]['bbox'], results_json[j]['bbox'], padding):
                    if results_json[i]['score'] > results_json[j]['score']:
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[j]['label']} with score {results_json[j]['score']} (there's a similar bbox with higher score)")
                        results_json.pop(j)
                    else:
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[i]['label']} with score {results_json[i]['score']} (there's a similar bbox with higher score)")
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

            if not (width >= image_width * config.get(LOCATION_LARGE_BBOX_RATIO) and
                    height >= image_height * config.get(LOCATION_LARGE_BBOX_RATIO)):
                filtered_results.append(result)
            elif verbose:
                print(f"{self.STR_PREFIX} Discarded: {result['label']} with score {result['score']} (bbox is too large)")
        results_json = filtered_results

        # 3. Discard bounding boxes with the same label and one fully contained in the other (discards the bigger one)
        i = 0
        while i < len(results_json):
            j = i + 1
            while j < len(results_json):
                if results_json[i]['label'] == results_json[j]['label']:
                    if __is_bbox_contained(results_json[i]['bbox'], results_json[j]['bbox'], padding):
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[j]['label']} with score {results_json[j]['score']} (contained another bbox with the same label)")
                        results_json.pop(j)
                        continue
                    elif __is_bbox_contained(results_json[j]['bbox'], results_json[i]['bbox'], padding):
                        if verbose:
                            print(f"{self.STR_PREFIX} Discarded: {results_json[i]['label']} with score {results_json[i]['score']} (contained another bbox with the same label)")
                        results_json.pop(i)
                        i -= 1
                        break
                j += 1
            i += 1

        return results_json


    def __filter_labels(self, results: Dict, input_tags: List[str]) -> dict:
        """
        Filters the results based on the label coincidence with the tagging stage.
        """
        for result in results:
            current_label = result["label"]
            substrings = []

            # Check if the label contains any of the input tags
            for tag in input_tags:
                if tag in current_label and tag != current_label:
                    substrings.append(tag)

            if substrings:
                # Replace the label with the shortest substring (based on the number of words and characters)
                best_substring = min(substrings, key=lambda s: (len(s.split()), len(s)))
                print(f"{self.STR_PREFIX} Replaced label: {current_label} with {best_substring}")
                result["label"] = best_substring

        return results

    def __draw_bounding_boxes(
            self,
            results: Dict,
            fixed_width: int = 800,
            show_padding: bool = False
    ) -> Image:
        """
        Draws bounding boxes around the detected objects in the image, scaling to a fixed width.
        """
        # Resize image to fixed width while maintaining aspect ratio
        original_width, original_height = self.input_image.size
        scale = fixed_width / original_width
        new_height = int(original_height * scale)
        image = self.input_image.resize((fixed_width, new_height), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # Draw bounding boxes for each detected object
        for obj in results:
            label = obj.get("label", "unknown")
            score = obj.get("score", 0.0)
            bbox = obj.get("bbox", {})

            # Extract and scale bounding box coordinates
            x_min = float(bbox.get("x_min", 0)) * scale
            y_min = float(bbox.get("y_min", 0)) * scale
            x_max = float(bbox.get("x_max", 0)) * scale
            y_max = float(bbox.get("y_max", 0)) * scale

            # Draw the bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=4)

            # Draw the label and score
            text = f"{label}: {score:.2f}"
            draw.text((x_min + 7, y_min + 3), text, fill="red", font=font)

        # Draw reference padding rectangle if show_padding is True
        if show_padding:
            _, image_height = image.size
            rect_width = float(original_width * config.get(LOCATION_PADDING_RATIO) * scale)
            rect_height = 30

            rect_x_min = 0
            rect_y_min = image_height - rect_height
            rect_x_max = rect_width
            rect_y_max = image_height

            draw.rectangle([rect_x_min, rect_y_min, rect_x_max, rect_y_max], outline="green", fill="green", width=2)

            # Draw the "padding" text
            text = "padding"
            text_x = rect_x_min + 5
            text_y = rect_y_min - 30
            draw.text((text_x, text_y), text, fill="green", font=font)

        return image


    def execute_location(self) -> List[Dict]:
        """
        Execute the Location process.

        This method will be called by the execute method in the subclasses.
        """
        # Convert the tags JSON text to a model prompt
        text = self.json_to_model_prompt(self.input_tags)

        # Subclass method to locate bounding boxes
        results = self.locate_bboxes(text)

        # Subclass method to convert the results to a JSON dict list
        results_json = self.model_results_to_json(results)

        # Filter the results based on the confidence threshold
        results_json = self.__filter_confidence(results_json)

        # Filter the results based on bounding box properties
        results_json = self.__filter_bbox(results_json, self.input_image.width, self.input_image.height)

        # Filter the results based on label coincidence with the tagging stage
        results_json = self.__filter_labels(results_json, self.input_tags)

        print(f"{self.STR_PREFIX} JSON results:\n\n{json.dumps(results_json, indent=4)}")
        
        return results_json

    @abstractmethod
    def json_to_model_prompt(self, tags: List[str]) -> str:
        raise NotImplementedError("json_to_model_prompt method must be implemented in subclasses.")
    
    @abstractmethod
    def locate_bboxes(self, text: str) -> Dict:
        raise NotImplementedError("locate_bboxes method must be implemented in subclasses.")

    @abstractmethod
    def model_results_to_json(self, results: Dict) -> Dict:
        raise NotImplementedError("model_results_to_json method must be implemented in subclasses.")

    # Override from ILocationStrategy
    def save_outputs(self, location: Dict) -> None:
        if config.get(SAVE_FILES):
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_location_json(location, timestamp)
            self.save_location_image(location, timestamp)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Location output was not saved.")

    # Override from ILocationStrategy
    def save_location_json(self, location: Dict, timestamp: str) -> None:
        if config.get(SAVE_FILES):
            # Prepare JSON output file
            output_filename = f"location_{timestamp}_{self.ALIAS}.json"
            output_file = os.path.join(OUTPUT_LOCATION_DIR, output_filename)

            # Save the results to a JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(location, f, indent=4)
            
            print(f"{self.STR_PREFIX} Bounding box instance location JSON results saved to: {output_file}")

    # Override from ILocationStrategy
    def save_location_image(self, location: Dict, timestamp: str) -> None:
        if config.get(SAVE_FILES):
            # Prepare JPG output file
            output_filename = f"location_{timestamp}_{self.ALIAS}.jpg"
            output_file = os.path.join(OUTPUT_LOCATION_DIR, output_filename)

            # Draw bounding boxes around the detected objects
            results_image = self.__draw_bounding_boxes(results=location, show_padding=True)

            # Save the image with bounding boxes
            results_image.save(output_file)
            print(f"{self.STR_PREFIX} Bounding box location image saved to: {output_file}")
