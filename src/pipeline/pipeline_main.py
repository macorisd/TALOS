import time
import argparse
from typing import List, Tuple, Union, Optional

from pipeline.config.config import (
    ConfigSingleton,
    PIPELINE_TAGGING,
    PIPELINE_LOCATION,
    PIPELINE_SEGMENTATION
)
from utils.print_utils import print_purple
from pipeline.factory.factory import StrategyFactory


class PipelineTALOS:
    def __init__(self, config_file_name: str = 'config.json'):
        print_purple("\n[PIPELINE] Loading configuration parameters...")
        
        tagging_method, location_method, segmentation_method = self.load_config(config_file_name)

        print_purple("\n[PIPELINE] Loading models...")
        
        self.set_tagging_strategy(tagging_method)
        self.set_location_strategy(location_method)
        self.set_segmentation_strategy(segmentation_method)
        
        print_purple("\n[PIPELINE] All models loaded successfully.")
    
    def load_config(self, config_file_name: str) -> Tuple[Union[str, List[str]], str, str]:
        self.config = ConfigSingleton(config_file=config_file_name) 

        return (
            self.config.get(PIPELINE_TAGGING),
            self.config.get(PIPELINE_LOCATION),
            self.config.get(PIPELINE_SEGMENTATION)
        )
    
    def set_tagging_strategy(self, tagging_method: Union[str, List[str]]):
        print_purple(f"\n[PIPELINE] Setting tagging strategy to: {tagging_method}")
        self.tagging_strategy = StrategyFactory.create_tagging_strategy(tagging_method)
        print_purple(f"\n[PIPELINE] Tagging strategy set to: {tagging_method}")

    def set_location_strategy(self, location_method: str):
        print_purple(f"\n[PIPELINE] Setting location strategy to: {location_method}")
        self.location_strategy = StrategyFactory.create_location_strategy(location_method)
        print_purple(f"\n[PIPELINE] Location strategy set to: {location_method}")
    
    def set_segmentation_strategy(self, segmentation_method: str):
        print_purple(f"\n[PIPELINE] Setting segmentation strategy to: {segmentation_method}")
        self.segmentation_strategy = StrategyFactory.create_segmentation_strategy(segmentation_method)
        print_purple(f"\n[PIPELINE] Segmentation strategy set to: {segmentation_method}")

    def run(self, input_image_names: Union[str, List[str]], iters: int = 1) -> Tuple[float, Optional[float]]:
        if isinstance(input_image_names, str):
            input_image_names = [input_image_names]

        total_time = 0
        total_runs = iters * len(input_image_names)

        for i in range(iters):
            if iters > 1:
                print_purple(f"\n[PIPELINE] Iteration {i+1}/{iters}")
            for j, image_name in enumerate(input_image_names):
                start_time = time.time()
                print_purple(f"\n[PIPELINE] Running pipeline for image {j+1}/{len(input_image_names)}: {image_name}...")

                # Tagging
                self.tagging_strategy.load_inputs(input_image_name=image_name)
                tags = self.tagging_strategy.execute()
                self.tagging_strategy.save_outputs(tags)

                # Location
                self.location_strategy.load_inputs(input_image_name=image_name, input_tags=tags)
                locations = self.location_strategy.execute()
                self.location_strategy.save_outputs(locations)

                # Segmentation
                self.segmentation_strategy.load_inputs(input_image_name=image_name, input_location=locations)
                segmentation_info, all_masks = self.segmentation_strategy.execute()
                self.segmentation_strategy.save_outputs(segmentation_info, all_masks)

                # Calculate execution time
                elapsed = time.time() - start_time
                total_time += elapsed
                print_purple(f"\n[PIPELINE] Finished in {elapsed:.2f} seconds.")

        if total_runs > 1:
            average_time = total_time / total_runs
            print_purple(f"\n[PIPELINE] Total execution time for {total_runs} executions: {total_time:.2f} seconds.")
            print_purple(f"\n[PIPELINE] Average execution time: {average_time:.2f} seconds.")
        else:
            average_time = None

        print_purple("\n[PIPELINE] All images processed successfully.")
        return total_time, average_time


def main(input_image_names: Union[str, List[str]], iters: int = 1, config_file_name: str = 'config.json'):
    if not input_image_names:
        raise ValueError("\n[PIPELINE] No input image names provided. Please provide a list of image names.")

    pipeline = PipelineTALOS(config_file_name=config_file_name)
    pipeline.run(input_image_names, iters=iters)


if __name__ == "__main__":
    # ArgumentParser to handle command line arguments
    parser = argparse.ArgumentParser(description="Run the TALOS pipeline on specified images.")
    
    # Add an argument for input images
    parser.add_argument(
        "-img",
        "--input_images",
        nargs='*', # Zero or more arguments
        default=['desk.jpg'], # Default image if none are provided
        help="One or more input image names (e.g., image1.png image2.jpg). Defaults to ['desk.jpg'] if not specified. Images must be located in the 'input_images' directory.",
        metavar="IMAGE_NAME"
    )
    
    # Add an argument for iterations
    parser.add_argument(
        "-iters",
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run the pipeline for each image (default: 1).",
        metavar="NUM_ITERATIONS"
    )

    # Add an argument for the configuration file
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        default='config.json', # Default configuration file
        help="Name of the configuration file to use (e.g., config.json, config2.json). Defaults to 'config.json'. Must be located in the config directory.",
        metavar="CONFIG_FILENAME"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(input_image_names=args.input_images, iters=args.iterations, config_file_name=args.config_file)
