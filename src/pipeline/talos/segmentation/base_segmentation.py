from abc import abstractmethod
import json
import os
from typing import Dict, List, Tuple
from PIL import Image
import cv2
import numpy as np

from pipeline.strategy.strategy import ISegmentationStrategy
from pipeline.config.config import (
    ConfigSingleton,
    SAVE_INTERMEDIATE_FILES,
    SAVE_SEGMENTATION_FILES
)
from pipeline.config.paths import (
    INPUT_IMAGES_DIR,
    OUTPUT_LOCATION_DIR,
    OUTPUT_SEGMENTATION_DIR
)
from pipeline.common.file_saving import FileSaving

class BaseSegmenter(ISegmentationStrategy):
    """
    [Segmentation]

    Base class for segmentation strategies.
    """

    def __init__(self):
        """
        Initialize the base segmenter.
        """
        global config
        config = ConfigSingleton()

    # Override from ISegmentationStrategy
    def load_inputs(
        self,
        input_image_name: str = None,
        input_image: np.ndarray = None,
        input_location: List[Dict] = None
    ) -> None:
        """
        Load the Segmentation inputs.
        """
        if input_image_name is None and input_image is None:
            raise ValueError(f"{self.STR_PREFIX} Either input_image_name or input_image must be provided.")
        
        if input_image is not None:
            self.set_image(input_image)
        else:
            self.load_image(input_image_name)

        # Load the input location information
        self.load_location(input_location)

    # Override from ISegmentationStrategy
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image for the Segmentation stage.
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
        
        if config.get(SAVE_SEGMENTATION_FILES):
            # Save image information in segmentation_info
            self.segmentation_info = {
                "image_name": input_image_name,
                "width": self.input_image.width,
                "height": self.input_image.height,
                "detections": []
            }
        else:
            self.segmentation_info = {}
        
        print("Done.")

    # Override from ISegmentationStrategy
    def set_image(self, input_image: np.ndarray) -> None:
        """
        Set the input image.
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.input_image = Image.fromarray(rgb_image)

        if config.get(SAVE_SEGMENTATION_FILES):
            # Save image information in segmentation_info
            self.segmentation_info = {
                "width": self.input_image.width,
                "height": self.input_image.height,
                "detections": []
            }
        else:
            self.segmentation_info = {}

    # Override from ISegmentationStrategy
    def load_location(self, input_location: List[Dict] = None) -> None:
        """
        Load the input location information (output from the Location stage) for the Segmentation stage.
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
    
    @abstractmethod
    def execute(self) -> List[str]:
        raise NotImplementedError("The execute method must be implemented in the subclass.")

    def execute_segmentation(self) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Execute the Segmentation process.

        This method generates binary masks for the input image based on the bounding boxes from the input location.

        This method will be called by the execute method in the subclasses.
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

            if config.get(SAVE_SEGMENTATION_FILES):
                # Append the location information to segmentation_info
                self.segmentation_info["detections"].append({
                    "id": i + 1,
                    "label": instance.get("label", "unknown"),
                    "score": instance.get("score", 0),
                    "bbox": bbox_coords
                })

            # Generate the binary mask for the current instance
            binary_mask = self.generate_mask(bbox_coords)
            all_masks.append(binary_mask)

        print("Done.")

        return self.segmentation_info, all_masks

    @abstractmethod
    def generate_mask(self, bbox_coords: List[float]) -> np.ndarray:
        raise NotImplementedError("The generate_mask method must be implemented in the subclass.")
    
    # Override from ISegmentationStrategy
    def save_outputs(self, segmentation_info: Dict, all_masks: List[np.ndarray]) -> None:
        """
        Save the segmentation outputs to files.
        """
        if config.get(SAVE_INTERMEDIATE_FILES) or config.get(SAVE_SEGMENTATION_FILES):
            self.output_timestamped_dir = FileSaving.create_output_directory(
                parent_dir=OUTPUT_SEGMENTATION_DIR,
                output_name="segmentation",
                alias=self.ALIAS
            )
            
            if config.get(SAVE_SEGMENTATION_FILES):
                # Save instance detection information to JSON
                self.save_detections_json(segmentation_info)

                # Save segmentation masks to .npz files
                self.save_masks_npz(all_masks)

            if config.get(SAVE_INTERMEDIATE_FILES):
                # Save segmentation masks as images
                self.save_mask_images(all_masks)

                # Save highlighted images with masks and labels
                self.save_mask_highlighted_images(all_masks)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Segmentation output was not saved.")
    
    # Override from ISegmentationStrategy
    def save_detections_json(self, segmentation_info: Dict) -> None:
        """
        Save the instance detection information to a JSON file.
        """
        if config.get(SAVE_SEGMENTATION_FILES):
            # Output JSON file path
            output_json_path = os.path.join(
                self.output_timestamped_dir,
                f"detections_info.json"
            )

            # Save the instance detections information to a JSON file
            with open(output_json_path, "w") as f:
                json.dump(segmentation_info, f, indent=4)
            
            print(f"{self.STR_PREFIX} Instance detection information JSON saved to: {output_json_path}")
    
    # Override from ISegmentationStrategy
    def save_masks_npz(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks to .npz files.
        """
        if config.get(SAVE_SEGMENTATION_FILES):
            [self.__save_segmentation_mask(mask, i+1) for i, mask in enumerate(masks)]

    def __save_segmentation_mask(self, mask: np.ndarray, idx: int) -> None:
        """
        Save a single segmentation mask to a .npz file.
        """
        # Output mask file path
        output_mask_path = os.path.join(
            self.output_timestamped_dir,
            f"segmentation_{self.ALIAS}_mask_{str(idx)}.npz"
        )

        # Save the mask as a .npz file
        np.savez_compressed(output_mask_path, mask=mask)
        
        print(f"{self.STR_PREFIX} Segmentation mask saved to: {output_mask_path}")

    # Override from ISegmentationStrategy
    def save_mask_images(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks as images.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            [self.__save_segmentation_image(mask, i+1) for i, mask in enumerate(masks)]
    
    def __save_segmentation_image(self, mask: np.ndarray, idx: int) -> None:
        """
        Save a single segmentation mask as an image.
        """
        # Output image file path
        output_image_path = os.path.join(
            self.output_timestamped_dir,
            f"segmentation_{self.ALIAS}_mask_{str(idx)}.png"
        )

        # Convert mask to uint8 and save as PNG
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(output_image_path)
        
        print(f"{self.STR_PREFIX} Segmentation mask image saved to: {output_image_path}")
    
    # Override from ISegmentationStrategy
    def save_mask_highlighted_images(self, masks: List[np.ndarray]) -> None:
        """
        Save the segmentation masks as images with highlighted segments.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            # Convert PIL image to OpenCV format (numpy array in BGR)
            image_bgr = cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2BGR)

            for i, mask in enumerate(masks):
                label = self.input_location[i].get("label", "unknown")
                overlayed_image = self.__highlight_image_mask(image_bgr, mask, label=label)

                # Output image file path
                output_image_path = os.path.join(
                    self.output_timestamped_dir,
                    f"segmentation_{self.ALIAS}_mask_{str(i+1)}_highlighted.png"
                )

                # Save image using OpenCV
                cv2.imwrite(output_image_path, overlayed_image)

                print(f"{self.STR_PREFIX} Highlighted segmentation image saved to: {output_image_path}")

    def __highlight_image_mask(self, image, mask, label="unknown", color=(0, 255, 0), alpha=0.5, fixed_width=800) -> np.ndarray:
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
