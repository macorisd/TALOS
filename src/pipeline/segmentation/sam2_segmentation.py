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

    STR_PREFIX = "\n[SEGMENTATION | SAM2]"

    def __init__(
        self,        
        sam2_model_name: str = "facebook/sam2-hiera-large",
        save_files_jpg: bool = True,
        save_files_npy: bool = False,        
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"{self.STR_PREFIX} Initializing SAM2 instance segmenter...", end=" ")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.save_files_jpg = save_files_jpg
        self.save_files_npy = save_files_npy
        self.timeout = timeout if timeout > 0 else 120

        # Load SAM2 predictor
        self.predictor = SAM2ImagePredictor.from_pretrained(sam2_model_name)

        if save_files_jpg or save_files_npy:
            # Output segments directory path
            self.output_segments_dir = os.path.join(
                self.script_dir,
                "..",
                "output_segments"
            )

            # Create the output directory if it does not exist
            os.makedirs(self.output_segments_dir, exist_ok=True)
        
        print("Done.")
                
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
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.")

        print("Done.")

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
                raise FileNotFoundError(f"{self.STR_PREFIX} No .json files found in {input_bbox_location_dir}")
            
            # Select the most recently modified .json file
            latest_json_path = max(json_files, key=os.path.getmtime)
            
            # Extract the filename (without the full path)
            filename = os.path.basename(latest_json_path)
            print("Most recent .json file:", filename, end="... ")

            # Read the content of the file
            with open(latest_json_path, "r") as f:
                self.input_bbox_location = json.load(f)

        print("Done.")

    def highlighted_segment_image(self, image, mask, label="unknown", color=(0, 255, 0), alpha=0.5):
        """
        Overlay a segmentation mask on the image and add a label.
        """
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_bool = mask.astype(bool)  # Ensure mask is boolean
        mask_overlay[mask_bool] = color
        
        # Blend original image with mask overlay
        overlayed_image = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
        
        # Add label text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x, text_y = 10, 10 + text_size[1]  # Position in top-left corner
        
        # Text background (black rectangle)
        cv2.rectangle(
            overlayed_image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        # Overlay text
        cv2.putText(overlayed_image, label, (text_x, text_y), font, font_scale, color, thickness)

        # Convert to BGR
        # cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
        
        return overlayed_image


    def build_path_npy(self, output_dir: str, idx: int) -> str:
        """
        Build the path for the output .npy file.
        """
        output_npy_path = os.path.join(
            output_dir,
            f"segment_{idx}_mask.npy"
        )
        return output_npy_path
    
    def build_path_jpg(self, output_dir: str, idx: int) -> str:
        """
        Build the path for the output .jpg file.
        """
        output_jpg_path = os.path.join(
            output_dir,
            f"segment_{idx}.jpg"
        )
        return output_jpg_path
    
    def run(self):
        """
        Perform instance segmentation using SAM2 and the provided bounding boxes.
        """
        print(f"{self.STR_PREFIX} Running SAM2 instance segmentation...")

        if self.save_files_jpg or self.save_files_npy:
                # Prepare timestamp
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

                # Output timestamped directory path
                base_output_timestamped_segments_dir = os.path.join(
                    self.output_segments_dir,
                    f"segmentation_sam2_{timestamp}"
                )

                # Ensure the output directory is unique
                output_timestamped_segments_dir = base_output_timestamped_segments_dir
                counter = 1

                while os.path.exists(output_timestamped_segments_dir):
                    output_timestamped_segments_dir = f"{base_output_timestamped_segments_dir}_{counter}"
                    counter += 1

                # Create the unique timestamped output directory 
                os.makedirs(output_timestamped_segments_dir)

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
                output_image_path = self.build_path_jpg(output_dir=output_timestamped_segments_dir, idx=i)
                
                highlighted_segment_image = self.highlighted_segment_image(
                    self.input_image, 
                    best_mask, 
                    label=instance.get("label", "unknown")
                )
                
                cv2.imwrite(
                    output_image_path,
                    cv2.cvtColor(highlighted_segment_image, cv2.COLOR_RGB2BGR)
                )
                
                print(f"{self.STR_PREFIX} Segmented image for instance {i} saved at: {output_image_path}")

            # Save the segmentation results to a JSON file if save_files_npy is True
            if self.save_files_npy:
                output_npy_path = self.build_path_npy(output_dir=output_timestamped_segments_dir, idx=i)
                np.save(output_npy_path, best_mask)
                print(f"{self.STR_PREFIX} Segmented mask for instance {i} saved at: {output_npy_path}")
    
def main():
    segmenter = Sam2Segmenter()
    segmenter.load_image("desk.jpg")
    segmenter.load_bbox_location()

    segmenter.run()

if __name__ == "__main__":
    main()