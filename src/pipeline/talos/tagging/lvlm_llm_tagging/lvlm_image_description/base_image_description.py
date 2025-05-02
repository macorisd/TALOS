from abc import abstractmethod
import os
import time
from typing import List

from strategy.strategy import ITaggingLvlmStrategy
from config.config import (
    config,
    SAVE_FILES,
    TAGGING_LVLM_ITERS,
    TAGGING_LVLM_TIMEOUT
)
from config.paths import (
    INPUT_IMAGES_DIR,
    OUTPUT_DESCRIPTIONS_DIR
)


class BaseLvlmImageDescriptor(ITaggingLvlmStrategy):
    """
    [Tagging -> LVLM + LLM -> LVLM Image Description]
    
    Base class for LVLM image descriptors for the Tagging stage with the LVLM + LLM method.
    """

    def __init__(
        self,
        prompt: str = "Describe the image."
    ):
        """
        Initialize the base LVLM image descriptor.
        """
        # Variables
        self.prompt = prompt

        if config.get(SAVE_FILES):
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_DESCRIPTIONS_DIR, exist_ok=True)
            self.create_output_directory()

    def load_inputs(self, input_image_name: str) -> None:
        """
        Load the LVLM Image Description inputs.
        """
        self.load_image(input_image_name)

    # Override
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image's path.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ")
        self.input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        if not os.path.isfile(self.input_image_path):
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {self.input_image_name} was not found.")
        print("Done.")


    @abstractmethod
    def execute(self) -> List[str]:
        pass


    def create_output_directory(self) -> None:
        """
        Create the output directory for saving current descriptions.
        """
        if config.get(SAVE_FILES):
            # Prepare timestamp
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Output timestamped directory path
            base_output_timestamped_dir = os.path.join(
                OUTPUT_DESCRIPTIONS_DIR,
                f"description_{self.ALIAS}_{timestamp}"
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
    

    def execute_image_description(self) -> List[str]:
        """
        Iteratively prompt the LVLM model to generate descriptions for the input image.
        """
        descriptions = [""] * config.get(TAGGING_LVLM_ITERS)
        start_time = time.time()  # Start timer for timeout

        for i in range(config.get(TAGGING_LVLM_ITERS)):
            if config.get(TAGGING_LVLM_ITERS) > 1:
                print(f"{self.STR_PREFIX} Iteration {i + 1}/{config.get(TAGGING_LVLM_ITERS)}...")

            # Describe the image
            while time.time() - start_time < config.get(TAGGING_LVLM_TIMEOUT):
                descriptions[i] = self.chat_lvlm()
                if descriptions[i].strip(): # Not empty
                    break
                else:
                    print(f"{self.STR_PREFIX} The description is empty. Trying again...")
            else:
                raise TimeoutError(f"{self.STR_PREFIX} Timeout of {config.get(TAGGING_LVLM_TIMEOUT)} seconds reached without receiving a valid description.")

            # Print the description
            print(f"{self.STR_PREFIX} Image description:\n\n" + descriptions[i])
        
        return descriptions

    def save_outputs(self, descriptions: List[str]) -> None:
        if config.get(SAVE_FILES):
            self.save_descriptions(descriptions)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Image description output was not saved.")

    # Override
    def save_descriptions(self, descriptions: List[str]) -> None:
        """
        Save the generated descriptions to text files.
        """
        if config.get(SAVE_FILES):
            for i in range(len(descriptions)):
                output_filename = f"description_{i+1}.txt" if config.get(TAGGING_LVLM_ITERS) > 1 else "description.txt"
                output_file = os.path.join(self.output_timestamped_dir, output_filename)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(descriptions[i])
                print(f"{self.STR_PREFIX} Image description saved to: {output_file}")

    
    @abstractmethod
    def chat_lvlm(self) -> str:
        pass
