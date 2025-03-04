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
        save_files_jpg: bool = True,
        save_files_npy: bool = True,        
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"\n{self.STR_PREFIX} Initializing SAM2 instance segmenter...\n")
        print(f"{self.STR_PREFIX} Input image name: {input_image_name}\n")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.save_files_jpg = save_files_jpg
        self.save_files_npy = save_files_npy
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

        if save_files_jpg or save_files_npy:
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

            if save_files_npy:
                self.output_files_npy = [
                    os.path.join(output_timestamped_segments_dir, f"segment_{i}_mask.npy")
                    for i in range(len(self.input_bbox_location))
                ]

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

        # Iterate over each instance in the input_bbox_location
        for i, instance in enumerate(self.input_bbox_location):                      
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

            # Save the segmented image if save_files_jpg is True
            if self.save_files_jpg:
                # Create an overlay image with the mask
                mask_overlay = np.zeros_like(self.input_image, dtype=np.uint8)
                color = (0, 255, 0) # Green color for the mask and text
                bg_color = (0, 0, 0)  # Black color for the text background
                mask_bool = best_mask.astype(bool)  # Convert mask to boolean
                mask_overlay[mask_bool] = color

                # Merge the original image with the segmentation
                alpha = 0.5  # Transparency level
                overlayed_image = cv2.addWeighted(self.input_image, 1 - alpha, mask_overlay, alpha, 0)

                # Add the label to the image
                label = instance.get("label", "unknown")
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 2
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x, text_y = 10, 10 + text_size[1]  # Position in top-left corner

                box_x, box_y = text_x - 5, text_y - text_size[1] - 5

                # Text background
                cv2.rectangle(
                    overlayed_image,
                    (box_x, box_y),
                    (text_x + text_size[0] + 5, text_y + 5),
                    bg_color,
                    -1
                )

                # Text
                cv2.putText(overlayed_image, label, (text_x, text_y), font, font_scale, color, thickness)

                # Save the overlayed image
                output_image_path = self.output_files_jpg[i]
                cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
                print(f"{self.STR_PREFIX} Segmented image for instance {i} saved at: {output_image_path}")

            # Save the segmentation results to a JSON file if save_files_npy is True
            if self.save_files_npy:
                output_npy_path = self.output_files_npy[i]
                np.save(output_npy_path, best_mask)
                print(f"{self.STR_PREFIX} Segmented mask for instance {i} saved at: {output_npy_path}")
    
def main():
    segmenter = Sam2Segmenter(
        input_image_name="279.jpg"
    )

    segmenter.segment_image()

if __name__ == "__main__":
    main()