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
        save_files_jpg: bool = True,
        save_files_npy: bool = True,        
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"\n{self.STR_PREFIX} Initializing SAM2 instance segmenter...", end=" ")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.save_files_jpg = save_files_jpg
        self.save_files_npy = save_files_npy
        self.timeout = timeout

        # Load SAM2 predictor
        self.predictor = SAM2ImagePredictor.from_pretrained(sam2_model_name)

        if save_files_jpg or save_files_npy:
            # Output segments directory path
            output_segments_dir = os.path.join(
                self.script_dir, 
                "output_segments"
            )

            # Create the output directory if it does not exist
            os.makedirs(output_segments_dir, exist_ok=True)

            # Prepare timestamped output files
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Output timestamped directory path
            self.output_timestamped_segments_dir = os.path.join(
                output_segments_dir,
                f"segmentation_sam2_{timestamp}"
            )

            # Create the timestamped output directory 
            os.makedirs(self.output_timestamped_segments_dir)
        
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
            input_image_bgr = cv2.imread(input_image_path)
            self.input_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.\n")

        print("Done.\n")

    def load_bbox_location(self, pipeline_bbox_location: dict = None) -> None:
        print(f"{self.STR_PREFIX} Loading input bounding box location information...", end=" ")

        # If pipeline_bbox_location is provided, use it
        if pipeline_bbox_location is not None:
            self.input_bbox_location = pipeline_bbox_location

        # Otherwise, read the most recent .json file from output_location
        else:
            # Input instance bounding box information JSON directory path
            input_bbox_location_dir = os.path.join(
                self.script_dir, 
                "..",
                "location",
                "output_location"
            )

            # Gather all .json files in input_bbox_location_dir
            json_files = [
                os.path.join(input_bbox_location_dir, f)
                for f in os.listdir(input_bbox_location_dir)
                if f.endswith(".json")
            ]
            if not json_files:
                raise FileNotFoundError(f"\n{self.STR_PREFIX} No .json files found in {input_bbox_location_dir}")
            
            # Select the most recently modified .json file
            latest_json_path = max(json_files, key=os.path.getmtime)
            
            # Extract the filename (without the full path)
            filename = os.path.basename(latest_json_path)
            print("Most recent .json file:", filename, end="... ")

            # Read the content of the file
            with open(latest_json_path, "r") as f:
                self.input_bbox_location = json.load(f)

        print("Done.\n")

    def build_path_npy(self, idx: int) -> str:
        """
        Build the path for the output .npy file.
        """
        output_npy_path = os.path.join(
            self.output_timestamped_segments_dir,
            f"segment_{idx}_mask.npy"
        )
        return output_npy_path
    
    def build_path_jpg(self, idx: int) -> str:
        """
        Build the path for the output .jpg file.
        """
        output_jpg_path = os.path.join(
            self.output_timestamped_segments_dir,
            f"segment_{idx}.jpg"
        )
        return output_jpg_path
    
    def run(self):
        """
        Perform instance segmentation using SAM2 and the provided bounding boxes.
        """
        print(f"{self.STR_PREFIX} Running SAM2 instance segmentation...\n")

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
                output_image_path = self.build_path_jpg(idx=i)
                cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
                print(f"{self.STR_PREFIX} Segmented image for instance {i} saved at: {output_image_path}")

            # Save the segmentation results to a JSON file if save_files_npy is True
            if self.save_files_npy:
                output_npy_path = self.build_path_npy(idx=i)
                np.save(output_npy_path, best_mask)
                print(f"{self.STR_PREFIX} Segmented mask for instance {i} saved at: {output_npy_path}")
    
def main():
    segmenter = Sam2Segmenter()
    segmenter.load_image("desk.jpg")
    segmenter.load_bbox_location()

    segmenter.run()

if __name__ == "__main__":
    main()