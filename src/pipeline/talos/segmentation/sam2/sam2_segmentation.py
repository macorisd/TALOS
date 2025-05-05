from typing import List
import torch
import numpy as np

from sam2.sam2_image_predictor import SAM2ImagePredictor

from pipeline.talos.segmentation.base_segmentation import BaseSegmenter

class Sam2Segmenter(BaseSegmenter):
    """
    [Segmentation -> SAM2]

    Segmentation stage implementation that leverages the SAM2 model.
    """

    STR_PREFIX = "\n[SEGMENTATION | SAM2]" # Prefix for logging
    ALIAS = "sam2" # Alias for filenames

    def __init__(
        self,
        sam2_model_name: str = "facebook/sam2-hiera-large"
    ):
        """
        Initialize the SAM2 segmenter.
        """
        print(f"{self.STR_PREFIX} Initializing SAM2 instance segmenter...", end=" ")

        # Initialize base class
        super().__init__()

        # Load the model
        self.predictor = SAM2ImagePredictor.from_pretrained(sam2_model_name)

        print("Done.")
    
    # Override
    def generate_mask(self, bbox_coords: List[int]) -> np.ndarray:
        """
        Generate a binary mask instance for the input image and the specified bbox coordinates,
        using the SAM2 model.
        """
        # Segmentation with SAM2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(self.input_image)
            masks, scores, _ = self.predictor.predict(box=[bbox_coords])

        # Get the best mask (the one with the highest score)
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]

        # Convert the mask to a binary format
        binary_mask = (best_mask > 0.5).astype(np.uint8)

        return binary_mask
    
    # Override
    def execute(self):
        print(f"{self.STR_PREFIX} Running instance segmentation with SAM2...", flush=True)

        segmentation_info, all_masks = self.execute_segmentation()
        return segmentation_info, all_masks

    
def main():
    """
    Main function to run the SAM2 segmenter.
    """
    segmenter = Sam2Segmenter()

    segmenter.load_inputs(input_image_name="avocado.jpeg")

    segmentation_info, all_masks = segmenter.execute()

    segmenter.save_outputs(segmentation_info, all_masks)


if __name__ == "__main__":
    main()