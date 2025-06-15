from abc import abstractmethod
import json
import os
import time
from typing import List
import numpy as np
from PIL import Image
import cv2

from pipeline.strategy.strategy import ITaggingStrategy
from pipeline.config.config import (
    ConfigSingleton,
    SAVE_INTERMEDIATE_FILES
)
from pipeline.config.paths import OUTPUT_TAGS_DIR


class BaseTagger(ITaggingStrategy):
    """
    [Tagging]
    
    Base class for Tagging implementations.
    """

    def __init__(self):
        """
        Initialize the base tagger.
        """
        global config
        config = ConfigSingleton()
        
        if config.get(SAVE_INTERMEDIATE_FILES):
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_TAGS_DIR, exist_ok=True)

    # Override from ITaggingStrategy
    def load_inputs(self, input_image_name: str = None, input_image: np.ndarray = None) -> None:
        """
        Load the Tagging inputs.
        """
        if input_image_name is None and input_image is None:
            raise ValueError(f"{self.STR_PREFIX} Either input_image_name or input_image must be provided.")
        
        if input_image is not None:
            self.set_image(input_image)
        else:
            self.load_image(input_image_name)

    @abstractmethod # from ITaggingStrategy
    def load_image(self, input_image_name: str) -> None:
        raise NotImplementedError("load_image method must be implemented in subclasses.")

    # Override from ITaggingStrategy
    def set_image(self, input_image: np.ndarray) -> None:
        """
        Set the input image.
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.input_image = Image.fromarray(rgb_image)

    @abstractmethod # from ITaggingStrategy
    def execute(self) -> List[str]:
        raise NotImplementedError("execute method must be implemented in subclasses.")

    # Override from ITaggingStrategy
    def save_outputs(self, tags: List[str]) -> None:
        """
        Save the Tagging outputs.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_tags(tags, timestamp)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Tagging output was not saved.")

    # Override from ITaggingStrategy
    def save_tags(self, tags: List[str], timestamp: str) -> None:
        """
        Save the Tagging output tags to a JSON file.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            # Prepare timestamped output file
            output_filename = f"tags_{timestamp}_{self.ALIAS}.json"
            output_file = os.path.join(OUTPUT_TAGS_DIR, output_filename)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tags, f, ensure_ascii=False, indent=4)
            print(f"{self.STR_PREFIX} Tags saved to: {output_file}")
