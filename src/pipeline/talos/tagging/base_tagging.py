from abc import abstractmethod
import json
import os
import time
from typing import List

from pipeline.strategy.strategy import ITaggingStrategy
from pipeline.config.config import (
    config,
    SAVE_FILES
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
        if config.get(SAVE_FILES):
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_TAGS_DIR, exist_ok=True)

    def load_inputs(self, input_image_name: str) -> None:
        """
        Load the Tagging inputs.
        """
        self.load_image(input_image_name)

    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass

    @abstractmethod
    def execute(self) -> List[str]:
        pass

    def save_outputs(self, tags: List[str]) -> None:
        if config.get(SAVE_FILES):
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_tags(tags, timestamp)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Tagging output was not saved.")

    # Override
    def save_tags(self, tags: List[str], timestamp: str) -> None:
        if config.get(SAVE_FILES):
            # Prepare timestamped output file
            output_filename = f"tags_{self.ALIAS}_{timestamp}.json"
            output_file = os.path.join(OUTPUT_TAGS_DIR, output_filename)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tags, f, ensure_ascii=False, indent=4)
            print(f"{self.STR_PREFIX} Tags saved to: {output_file}")
