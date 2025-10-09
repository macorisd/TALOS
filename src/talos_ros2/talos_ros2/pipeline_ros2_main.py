import warnings
warnings.filterwarnings('ignore', message='The value of the smallest subnormal')

import numpy as np
import time
from typing import Tuple, Dict, List

from pipeline.pipeline_main import PipelineTALOS

class PipelineTALOSRos2(PipelineTALOS):
    """
    TALOS pipeline for ROS2.
    """
    def __init__(self):
        super().__init__()
    
    # Override from PipelineTALOS
    def run(self, input_image: np.ndarray) -> Tuple[Dict, List[np.ndarray], float]:
        start_time = time.time()

        # Tagging
        self.tagging_strategy.load_inputs(input_image=input_image)
        tags = self.tagging_strategy.execute()
        self.tagging_strategy.save_outputs(tags)

        # Location
        self.location_strategy.load_inputs(input_image=input_image, input_tags=tags)
        locations = self.location_strategy.execute()
        self.location_strategy.save_outputs(locations)

        # Segmentation
        self.segmentation_strategy.load_inputs(input_image=input_image, input_location=locations)
        segmentation_info, all_masks = self.segmentation_strategy.execute()
        self.segmentation_strategy.save_outputs(segmentation_info, all_masks)

        # Calculate execution time
        total_time = time.time() - start_time

        return segmentation_info, all_masks, total_time

def main():
    input_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image for testing
    pipeline = PipelineTALOSRos2()

    segmentation_info, all_masks, total_time = pipeline.run(input_image)
    print(f"Segmentation Info: {segmentation_info}")

if __name__ == "__main__":
    main()
