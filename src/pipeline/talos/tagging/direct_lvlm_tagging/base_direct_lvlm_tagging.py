from abc import abstractmethod
import json
import os
import time
from typing import List
from PIL import Image

from pipeline.config.paths import (
    INPUT_IMAGES_DIR,
    TAGGING_DIRECT_LVLM_PROMPT
)
from pipeline.talos.tagging.base_tagging import BaseTagger
from common.large_model_tagging import LargeModelForTagging
from pipeline.config.config import (
    config,
    TAGGING_DIRECT_LVLM_TIMEOUT,
    TAGGING_DIRECT_LVLM_ITERS,
    TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS,
    TAGGING_DIRECT_LVLM_BANNED_WORDS
)


class BaseDirectLvlmTagger(BaseTagger, LargeModelForTagging):
    """
    [Tagging -> Direct Tagging with LVLM]

    Base class for Direct LVLM Tagging implementations.
    """

    def __init__(self):
        """
        Initialize the base direct LVLM tagger.
        """
        # Initialize base class
        super().__init__()

    # Override from ITaggingStrategy -> BaseTagger
    def load_inputs(self, input_image_name: str) -> None:
        """
        Load the Direct LVLM Tagging inputs.
        """
        super().load_inputs(input_image_name)
        self.load_prompt()
    
    # Override from ITaggingStrategy -> BaseTagger
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image for the Direct LVLM Tagging.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ", flush=True)

        # Input image path
        input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        # Load input image
        if os.path.isfile(input_image_path):
            self.input_image = Image.open(input_image_path)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.")
        
        print("Done.")

    def load_prompt(self) -> None:
        """
        Load the prompt for the Qwen model.
        """
        # Load the prompt from the file
        with open(TAGGING_DIRECT_LVLM_PROMPT, 'r') as file:
            self.prompt = file.read().strip()
    
    @abstractmethod # from ITaggingStrategy -> BaseTagger
    def execute(self) -> List[str]:
        raise NotImplementedError("execute method must be implemented in subclasses.")

    def execute_direct_lvlm_tagging(self) -> List[str]:
        """
        Execute the Direct Tagging with LVLM.

        This method will be called by the execute method in the subclasses.
        """
        start_time = time.time()
        correct_json = [None] * config.get(TAGGING_DIRECT_LVLM_ITERS)

        for i in range(config.get(TAGGING_DIRECT_LVLM_ITERS)):
            if config.get(TAGGING_DIRECT_LVLM_ITERS) > 1:
                print(f"{self.STR_PREFIX} Iteration {i + 1}/{config.get(TAGGING_DIRECT_LVLM_ITERS)}...")

            while time.time() - start_time < config.get(TAGGING_DIRECT_LVLM_TIMEOUT):
                # Chat with the LVLM
                response = self.chat_lvlm()

                print(f"{self.STR_PREFIX} LVLM response:\n\n", response + "\n")

                # Check if the response is in the correct format
                correct_json[i] = self.correct_response_format(response)

                if correct_json[i] is not None and len(correct_json[i]) > 0:
                    break
                else:
                    print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...")
            else:
                raise TimeoutError(f"{self.STR_PREFIX} Timeout of {config.get(TAGGING_DIRECT_LVLM_TIMEOUT)} seconds reached without receiving a correct response format.")

        # Merge the responses if there are multiple iterations
        if config.get(TAGGING_DIRECT_LVLM_ITERS) > 1:
            print(f"{self.STR_PREFIX} {TAGGING_DIRECT_LVLM_ITERS} iterations completed. Merging the responses...", end=" ", flush=True)
            final_json = self.response_fusion(correct_json)
        # Otherwise, use the single response
        else:
            final_json = correct_json[0]

        # Convert the final response to lowercase
        final_json = [element.lower() for element in final_json]

        # If self.banned_words has been loaded, remove banned words from the output
        if config.get(TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS) and config.get(TAGGING_DIRECT_LVLM_BANNED_WORDS) is not None:
            final_json = self.filter_banned_words(
                response=final_json, 
                banned_words=config.get(TAGGING_DIRECT_LVLM_BANNED_WORDS)
            )

        # Remove values with more than a certain number of words
        final_json = self.filter_long_values(final_json)

        # Remove words that contain a substring that is also in the dictionary
        final_json = self.filter_redundant_substrings(final_json)

        # Remove duplicate plural words
        final_json = self.filter_duplicate_plurals(final_json)

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4))
        return final_json

    @abstractmethod
    def chat_lvlm(self) -> str:
        raise NotImplementedError("chat_lvlm method must be implemented in subclasses.")
