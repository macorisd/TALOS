import time
from typing import List, Tuple, Union

from config.config import (
    ConfigSingleton,
    PIPELINE_TAGGING,
    PIPELINE_LOCATION,
    PIPELINE_SEGMENTATION
)
from utils.print_utils import print_purple
from factory.factory import StrategyFactory


class PipelineTALOS:
    def __init__(self):
        print_purple("\n[PIPELINE] Loading configuration parameters...")
        tagging_method, location_method, segmentation_method = self.load_config()

        print_purple("\n[PIPELINE] Loading models...")
        
        self.tagging_strategy = StrategyFactory.create_tagging_strategy(tagging_method)
        self.location_strategy = StrategyFactory.create_location_strategy(location_method)
        self.segmentation_strategy = StrategyFactory.create_segmentation_strategy(segmentation_method)
        
        print_purple("\n[PIPELINE] All models loaded successfully.")
    
    def load_config(self) -> Tuple[str, str, str]:
        self.config = ConfigSingleton()

        return (
            self.config.get(PIPELINE_TAGGING),
            self.config.get(PIPELINE_LOCATION),
            self.config.get(PIPELINE_SEGMENTATION)
        )


    def run(self, input_image_names: Union[str, List[str]], iters: int = 1) -> float:
        if isinstance(input_image_names, str):
            input_image_names = [input_image_names]

        total_time = 0
        total_runs = iters * len(input_image_names)

        for i in range(iters):
            if iters > 1:
                print_purple(f"\n[PIPELINE] Iteration {i + 1}/{iters}")
            for image_name in input_image_names:
                start_time = time.time()
                print_purple(f"\n[PIPELINE] Running pipeline for image: {image_name}...")

                # Tagging
                self.tagging_strategy.load_inputs(image_name)
                tags = self.tagging_strategy.execute()
                self.tagging_strategy.save_outputs(tags)

                # Location
                self.location_strategy.load_inputs(image_name, tags)
                locations = self.location_strategy.execute()
                self.location_strategy.save_outputs(locations)

                # Segmentation
                self.segmentation_strategy.load_inputs(image_name, locations)
                segmentation_info, all_masks = self.segmentation_strategy.execute()
                self.segmentation_strategy.save_outputs(segmentation_info, all_masks)

                # Calculate execution time
                elapsed = time.time() - start_time
                total_time += elapsed
                print_purple(f"\n[PIPELINE] Finished in {elapsed:.2f} seconds.")

        if total_runs > 1:
            average_time = total_time / total_runs
            print_purple(f"\n[PIPELINE] Average execution time: {average_time:.2f} seconds.")

        print_purple("\n[PIPELINE] All images processed successfully.")
        return total_time


def main(input_image_names: Union[str, List[str]], iters: int = 1):
    if not input_image_names:
        raise ValueError("\n[PIPELINE] No input image names provided. Please provide a list of image names.")

    pipeline = PipelineTALOS()
    pipeline.run(input_image_names, iters=iters)


if __name__ == "__main__":
    main(input_image_names=["desk.jpg"], iters=1)
