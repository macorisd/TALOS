import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2Segmenter:
    """
    A class to perform instance segmentation using the SAM2 model.
    """

    STR_PREFIX = "[SEGMENTATION | SAM2]"

    def __init__(
        self,        
        sam2_model_name: str = "facebook/sam2-hiera-large",
        input_image_name: str = "input_image.jpg",
        save_file_json: bool = True,
        save_files_jpg: bool = True,
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"\n{self.STR_PREFIX} Initializing SAM2 instance segmenter...\n")
        print(f"{self.STR_PREFIX} Input image name: {input_image_name}\n")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_file_json = save_file_json
        self.save_files_jpg = save_files_jpg
        self.timeout = timeout

        # Load SAM2 predictor
        self.predictor = SAM2ImagePredictor.from_pretrained(sam2_model_name)

        # Input image path
        input_image_path = os.path.join(
            self.script_dir, 
            "..",
            "input_images",
            input_image_name
        )
        
        # Load input image
        if os.path.isfile(input_image_path):
            input_image_bgr = cv2.imread(input_image_path)
            self.input_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.\n")
        
        # Input instance bounding box information JSON directory path
        input_bbox_location_dir = os.path.join(
            self.script_dir, 
            "..",
            "location",
            "output_location"
        )

        # Load instance bounding box information from the most recent .json file in input_bbox_location_dir
        self.input_bbox_location, bbox_location_filename = self.read_input_bbox_location(input_bbox_location_dir=input_bbox_location_dir)

        # Print input information
        print(f"{self.STR_PREFIX} Input image filename: {input_image_path}\n")
        print(f"{self.STR_PREFIX} Input instance bounding box information filename: {input_bbox_location_dir}/{bbox_location_filename}\n")

        # Output segments directory path
        output_segments_dir = os.path.join(
            self.script_dir, 
            "output_segments"
        )

        if save_files_jpg or save_file_json:
            # Create the output directory if it does not exist
            os.makedirs(output_segments_dir, exist_ok=True)

            # Prepare timestamped output files
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Output timestamped directory path
            output_timestamped_segments_dir = os.path.join(
                output_segments_dir,
                f"segmentation_sam2_{timestamp}"
            )

            # Create the timestamped output directory 
            os.makedirs(output_timestamped_segments_dir)

            if save_file_json:
                # Prepare JSON output file
                output_filename_json = f"segmentation_sam2_{timestamp}.json"
                self.output_file_json = os.path.join(output_timestamped_segments_dir, output_filename_json)

            if save_files_jpg:
                # Prepare JPG output files
                self.output_files_jpg = [
                    os.path.join(output_timestamped_segments_dir, f"segment_{i}.jpg")
                    for i in range(len(self.input_bbox_location))
                ]
                

    def read_input_bbox_location(self, input_bbox_location_dir: str) -> tuple[dict, str]:
        """
        Reads the instance bounding box information from the most recent .json file in the input_bbox_location_dir.
        """
        # Gather all .json files in input_bbox_location_dir
        json_files = [
            os.path.join(input_bbox_location_dir, f)
            for f in os.listdir(input_bbox_location_dir)
            if f.endswith(".json")
        ]
        if not json_files:
            raise FileNotFoundError(f"\n{self.STR_PREFIX} No .json files found in {input_bbox_location_dir}")
        
        # Load the most recent .json file
        json_files.sort(key=os.path.getmtime, reverse=True)
        bbox_location_filename = os.path.basename(json_files[0])
        with open(json_files[0], "r") as f:
            bbox_location = json.load(f)
        
        return bbox_location, bbox_location_filename
    
    def segment_image(self):
        """
        Perform instance segmentation using SAM2 and the provided bounding boxes.
        """
        print(f"{self.STR_PREFIX} Starting segmentation process...\n")

        # List to store segmentation results
        segmentation_results = []

        # Iterate over each instance in the input_bbox_location
        for i, instance in enumerate(self.input_bbox_location):
            label = instance.get("label", "unknown")            
            bbox = instance.get("bbox", {})
            
            # Extract bounding box coordinates
            x_min = int(bbox.get("x_min", 0))
            y_min = int(bbox.get("y_min", 0))
            x_max = int(bbox.get("x_max", 0))
            y_max = int(bbox.get("y_max", 0))
            
            # Define the bounding box in the format [x_min, y_min, x_max, y_max]
            bbox_coords = [x_min, y_min, x_max, y_max]

            # Segmentation with SAM2
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(self.input_image)
                masks, scores, _ = self.predictor.predict(box=[bbox_coords])

            # Get the best mask (the one with the highest score)
            best_mask_index = np.argmax(scores)
            best_mask = masks[best_mask_index]

            # Store the segmentation result
            segmentation_result = {
                "label": label,                
                "bbox": bbox,
                "mask": best_mask.tolist()  # Convert mask to list for JSON serialization
            }
            segmentation_results.append(segmentation_result)

            # Save the segmented image if save_files_jpg is True
            if self.save_files_jpg:
                # Create an overlay image with the mask
                mask_overlay = np.zeros_like(self.input_image, dtype=np.uint8)
                color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)  # Random color
                mask_bool = best_mask.astype(bool)  # Convert mask to boolean
                mask_overlay[mask_bool] = color

                # Merge the original image with the segmentation
                alpha = 0.5  # Transparency level
                overlayed_image = cv2.addWeighted(self.input_image, 1 - alpha, mask_overlay, alpha, 0)

                # Save the overlayed image
                output_image_path = self.output_files_jpg[i]
                cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
                print(f"{self.STR_PREFIX} Segmented image for instance {i} saved at: {output_image_path}")

        # Save the segmentation results to a JSON file if save_file_json is True
        if self.save_file_json:
            with open(self.output_file_json, "w") as f:
                json.dump(segmentation_results, f, indent=4)
                print(f"{self.STR_PREFIX} Segmentation results saved to: {self.output_file_json}")

        return segmentation_results
    
def main():
    segmenter = Sam2Segmenter(
        input_image_name="desk.jpg"
    )

    segmenter.segment_image()

if __name__ == "__main__":
    main()