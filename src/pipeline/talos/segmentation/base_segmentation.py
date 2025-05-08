from abc import abstractmethod
import json
import os
from typing import Dict, List, Tuple
from PIL import Image
import cv2
import numpy as np
import time

from pipeline.strategy.strategy import ISegmentationStrategy
from pipeline.config.config import (
    config,
    SAVE_FILES
)
from pipeline.config.paths import (
    INPUT_IMAGES_DIR,
    OUTPUT_LOCATION_DIR,
    OUTPUT_SEGMENTATION_DIR
)

class BaseSegmenter(ISegmentationStrategy):
    """
    [Segmentation]

    Base class for segmentation strategies.
    """

    STR_PREFIX = "\n[SEGMENTATION]"
    ALIAS = "base"

    def __init__(self):
        """
        Initialize the base segmenter.
        """
        if config.get(SAVE_FILES):
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_SEGMENTATION_DIR, exist_ok=True)
            self.create_output_directory()
    
    def load_inputs(self, input_image_name: str, input_location: List[Dict] = None) -> None:
        """
        Load the Segmentation inputs.
        """
        # Load the input image
        self.load_image(input_image_name)

        # Load the input location information
        self.load_location(input_location)

    # Override
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ", flush=True)

        # Input image path
        input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        # Load input image
        if os.path.isfile(input_image_path):
            self.input_image = Image.open(input_image_path)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.")
        
        if config.get(SAVE_FILES):
            # Save image information in segmentation_info
            self.segmentation_info = {
                "image_name": input_image_name,
                "width": self.input_image.width,
                "height": self.input_image.height,
                "detections": []
            }
        
        print("Done.")

    # Override
    def load_location(self, input_location: List[Dict] = None) -> None:
        """
        Load the input location information (output from the Location stage).
        """
        print(f"{self.STR_PREFIX} Loading input location information...", end=" ")

        # If input_location is provided, use it
        if input_location is not None:
            self.input_location = input_location

        # Otherwise, read the most recent .json file from output_location
        else:
            # Gather all .json files in OUTPUT_LOCATION_DIR
            json_files = [
                os.path.join(OUTPUT_LOCATION_DIR, f)
                for f in os.listdir(OUTPUT_LOCATION_DIR)
                if f.endswith(".json")
            ]
            if not json_files:
                raise FileNotFoundError(f"{self.STR_PREFIX} No .json files found in {OUTPUT_LOCATION_DIR}")

            # Select the most recently modified .json file
            latest_json_path = max(json_files, key=os.path.getmtime)

            # Extract the filename (without the full path)
            filename = os.path.basename(latest_json_path)
            print("Most recent .json file:", filename, end="... ")

            # Read the content of the file
            with open(latest_json_path, "r") as f:
                self.input_location = json.load(f)

        print("Done.")

    def execute_segmentation(self) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Generate binary masks for the input image based on the bounding boxes from the input location.
        """
        # List to store all masks
        all_masks = []

        # Iterate over each instance in the input_location
        for i, instance in enumerate(self.input_location):
            bbox = instance.get("bbox", {})

            # Extract bounding box coordinates
            x_min = float(bbox.get("x_min", 0))
            y_min = float(bbox.get("y_min", 0))
            x_max = float(bbox.get("x_max", 0))
            y_max = float(bbox.get("y_max", 0))

            # Define the bounding box in the format [x_min, y_min, x_max, y_max]
            bbox_coords = [x_min, y_min, x_max, y_max]

            if config.get(SAVE_FILES):
                # Append the location information to segmentation_info
                self.segmentation_info["detections"].append({
                    "id": i + 1,
                    "label": instance.get("label", "unknown"),
                    "bbox": bbox_coords
                })

            # Generate the binary mask for the current instance
            binary_mask = self.generate_mask(bbox_coords)
            all_masks.append(binary_mask)

        print("Done.")

        return self.segmentation_info, all_masks

    def create_output_directory(self) -> None:
        """
        Create the output directory for saving the current segmentation.
        """
        if config.get(SAVE_FILES):
            # Prepare timestamp
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Output timestamped directory path
            base_output_timestamped_dir = os.path.join(
                OUTPUT_SEGMENTATION_DIR,
                f"segmentation_{timestamp}_{self.ALIAS}"
            )

            # Ensure the output directory is unique
            output_timestamped_dir = base_output_timestamped_dir
            counter = 1

            while os.path.exists(output_timestamped_dir):
                output_timestamped_dir = f"{base_output_timestamped_dir}_{counter}"
                counter += 1
            
            self.output_timestamped_dir = output_timestamped_dir

            # Create the unique timestamped output directory
            os.makedirs(self.output_timestamped_dir)
    
    def save_outputs(self, segmentation_info: Dict, all_masks: List[np.ndarray]) -> None:
        if config.get(SAVE_FILES):
            # Save instance detection information to JSON
            self.save_detections_json(segmentation_info)

            # Save segmentation masks to .npz files
            self.save_segmentation_masks(all_masks)

            # Save segmentation masks as images
            self.save_segmentation_images(all_masks)

            # Save highlighted images with masks and labels
            self.save_segmentation_highlighted_images(all_masks)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Segmentation output was not saved.")
    
    def save_detections_json(self, segmentation_info: Dict) -> None:
        """
        Save the instance detection information to a JSON file.
        """
        if config.get(SAVE_FILES):
            # Output JSON file path
            output_json_path = os.path.join(
                self.output_timestamped_dir,
                f"detections_info.json"
            )

            # Save the instance detections information to a JSON file
            with open(output_json_path, "w") as f:
                json.dump(segmentation_info, f, indent=4)
            
            print(f"{self.STR_PREFIX} Instance detection information JSON saved to: {output_json_path}")
    
    def save_segmentation_masks(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks to .npz files.
        """
        if config.get(SAVE_FILES):
            [self.save_segmentation_mask(mask, i+1) for i, mask in enumerate(masks)]

    def save_segmentation_mask(self, mask: np.ndarray, idx: int) -> None:
        """
        Save the segmentation mask to a .npz file.
        """
        if config.get(SAVE_FILES):
            # Output mask file path
            output_mask_path = os.path.join(
                self.output_timestamped_dir,
                f"segmentation_{self.ALIAS}_mask_{str(idx)}.npz"
            )

            # Save the mask as a .npz file
            np.savez_compressed(output_mask_path, mask=mask)
            
            print(f"{self.STR_PREFIX} Segmentation mask saved to: {output_mask_path}")

    def save_segmentation_images(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks as images.
        """
        if config.get(SAVE_FILES):
            [self.save_segmentation_image(mask, i+1) for i, mask in enumerate(masks)]
    
    def save_segmentation_image(self, mask: np.ndarray, idx: int) -> None:
        """
        Save the segmentation mask as an image.
        """
        if config.get(SAVE_FILES):
            # Output image file path
            output_image_path = os.path.join(
                self.output_timestamped_dir,
                f"segmentation_{self.ALIAS}_mask_{str(idx)}.png"
            )

            # Convert mask to uint8 and save as PNG
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_image.save(output_image_path)
            
            print(f"{self.STR_PREFIX} Segmentation mask image saved to: {output_image_path}")
    
    def save_segmentation_highlighted_images(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks as images with highlighted segments.
        """
        if config.get(SAVE_FILES):
            # Convert PIL image to OpenCV format (numpy array in BGR)
            image_bgr = cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2BGR)

            for i, mask in enumerate(masks):
                label = self.input_location[i].get("label", "unknown")
                overlayed_image = self.highlight_image_mask(image_bgr, mask, label=label)

                # Output image file path
                output_image_path = os.path.join(
                    self.output_timestamped_dir,
                    f"segmentation_{self.ALIAS}_mask_{str(i+1)}_highlighted.png"
                )

                # Save image using OpenCV
                cv2.imwrite(output_image_path, overlayed_image)

                print(f"{self.STR_PREFIX} Highlighted segmentation image saved to: {output_image_path}")

    
    def highlight_image_mask(self, image, mask, label="unknown", color=(0, 255, 0), alpha=0.5, fixed_width=800) -> np.ndarray:
        """
        Overlay a segmentation mask on the image and add a label.
        The image and mask are resized to a fixed width, keeping aspect ratio, before overlaying the label.
        """
        # Resize the image to a fixed width, keeping aspect ratio
        aspect_ratio = image.shape[1] / float(image.shape[0])
        new_height = int(fixed_width / aspect_ratio)
        resized_image = cv2.resize(image, (fixed_width, new_height))

        # Resize the mask to match the resized image
        resized_mask = cv2.resize(mask.astype(np.uint8), (fixed_width, new_height))

        mask_overlay = np.zeros_like(resized_image, dtype=np.uint8)
        mask_bool = resized_mask.astype(bool)  # Ensure mask is boolean
        mask_overlay[mask_bool] = color

        # Blend original image with mask overlay
        overlayed_image = cv2.addWeighted(resized_image, 1 - alpha, mask_overlay, alpha, 0)

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

        return overlayed_image
    
    @abstractmethod
    def generate_mask(self, bbox_coords: List[float]) -> np.ndarray:
        pass