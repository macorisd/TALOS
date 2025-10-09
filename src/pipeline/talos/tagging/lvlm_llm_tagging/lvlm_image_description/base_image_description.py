from abc import abstractmethod
import os
import time
from typing import List
import numpy as np
import cv2
from PIL import Image

from pipeline.strategy.strategy import ITaggingLvlmStrategy
from pipeline.config.config import (
    ConfigSingleton,
    SAVE_INTERMEDIATE_FILES,
    TAGGING_LVLM_ITERS,
    TAGGING_LVLM_TIMEOUT
)
from pipeline.config.paths import (
    INPUT_IMAGES_DIR,
    OUTPUT_DESCRIPTIONS_DIR
)
from pipeline.common.file_saving import FileSaving

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
        global config
        config = ConfigSingleton()

        # Variables
        self.prompt = prompt

    # Override from ITaggingLvlmStrategy
    def load_inputs(self, input_image_name: str) -> None:
        """
        Load the LVLM Image Description inputs.
        """
        self.load_image(input_image_name)

    # Override from ITaggingLvlmStrategy
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image's path.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ")
        self.input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        if not os.path.isfile(self.input_image_path):
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {self.input_image_name} was not found.")
        print("Done.")
    
    # Override from ITaggingLvlmStrategy
    def set_image(self, input_image: np.ndarray) -> None:
        """
        Set the input image.
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.input_image = Image.fromarray(rgb_image)

    @abstractmethod # from ITaggingLvlmStrategy
    def execute(self) -> List[str]:
        raise NotImplementedError("execute method must be implemented in subclasses.")

    def execute_image_description(self) -> List[str]:
        """
        Execute the LVLM Image Description for the Tagging with LVLM + LLM.
        
        This method iteratively prompts the LVLM model to generate descriptions for the input image.

        This method will be called by the execute method in the subclasses.
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

    # Override from ITaggingLvlmStrategy
    def save_outputs(self, descriptions: List[str]) -> None:
        if config.get(SAVE_INTERMEDIATE_FILES):
            self.output_timestamped_dir = FileSaving.create_output_directory(
                parent_dir=OUTPUT_DESCRIPTIONS_DIR,
                output_name="description",
                alias=self.ALIAS
            )
            
            self.save_descriptions(descriptions)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Image description output was not saved.")

    # Override from ITaggingLvlmStrategy
    def save_descriptions(self, descriptions: List[str]) -> None:
        """
        Save the generated descriptions to text files.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            for i in range(len(descriptions)):
                output_filename = f"description_{i+1}.txt" if config.get(TAGGING_LVLM_ITERS) > 1 else "description.txt"
                output_file = os.path.join(self.output_timestamped_dir, output_filename)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(descriptions[i])
                print(f"{self.STR_PREFIX} Image description saved to: {output_file}")

    @abstractmethod
    def chat_lvlm(self) -> str:
        raise NotImplementedError("chat_lvlm method must be implemented in subclasses.")
