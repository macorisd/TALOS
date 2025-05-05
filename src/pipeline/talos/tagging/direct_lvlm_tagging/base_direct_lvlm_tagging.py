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
from pipeline.config.config import (
    config,
    TAGGING_DIRECT_LVLM_TIMEOUT,
    TAGGING_DIRECT_LVLM_ITERS,
    TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS,
    TAGGING_DIRECT_LVLM_BANNED_WORDS
)


class BaseDirectLvlmTagger(BaseTagger):
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

    # Override
    def load_inputs(self, input_image_name: str) -> None:
        """
        Load the Tagging inputs.
        """
        self.load_image(input_image_name)
        self.load_prompt()
    
    # Override
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image.
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
    
    def correct_response_format(self, text: str) -> List[str] | None:
        """
        Checks if there is a substring within the given text that starts with '[' and ends with ']'
        and follows the exact format:

        ["phrase", "phrase", "phrase", ...]

        If the format is correct, returns the JSON substring as a List[str]. Otherwise, returns None.
        """
        # Find indices of the last '[' and the last ']'
        start_index = text.rfind('[')
        end_index = text.rfind(']')

        # If we can't find a proper pair of brackets, return None
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return None

        # Extract the substring that includes the brackets
        substring = text[start_index:end_index + 1]

        # If there's a "/" character in the substring, it's not a valid JSON
        if "/" in substring:
            return None

        # Delete all backslashes and underscores from the substring
        substring = substring.replace("\\", "")
        substring = substring.replace("_", " ")

        # Attempt to parse the substring as JSON
        try:
            parsed_data = json.loads(substring)
        except json.JSONDecodeError:
            return None

        # The parsed data must be a list
        if not isinstance(parsed_data, list):
            return None
        
        # Check each element in the list
        for element in parsed_data:
            # Each element must be a string
            if not isinstance(element, str):
                return None

        return parsed_data

    def response_fusion(self, responses: List[List[str]]) -> List[str]:
        """
        Merges the responses of multiple iterations into a single list,
        ensuring unique string values in order.
        """
        print(f"{self.STR_PREFIX} {config.get(TAGGING_DIRECT_LVLM_ITERS)} iterations completed. Merging the responses...", end=" ", flush=True)

        final_response = []
        unique_values = set()

        # Collect all unique values from the lists
        for response in responses:
            for value in response:
                if value not in unique_values:
                    unique_values.add(value)
                    final_response.append(value)

        print(f"Done. Merged response substring:\n\n", json.dumps(final_response, indent=4))

        return final_response

    def filter_banned_words(self, response: List[str], ban_exact_word: bool = False) -> List[str]:
        print(f"{self.STR_PREFIX} Removing banned words...", flush=True)

        banned_words = set(config.get(TAGGING_DIRECT_LVLM_BANNED_WORDS))

        for banned_word in banned_words:
            for element in response:
                if (ban_exact_word and banned_word == element) or (not ban_exact_word and banned_word in element):
                    print(f"{self.STR_PREFIX} Discarded keyword: {element} (banned word: {banned_word})")
                    response = [value for value in response if banned_word not in value]
                    break

        return response
    
    def filter_long_values(self, response: List[str], n_words: int = 2) -> List[str]:
        print(f"{self.STR_PREFIX} Removing values with more than {n_words} words...", flush=True)

        result = []
    
        for element in response:
            # Check if the element has less than or equal to n_words
            if len(element.split()) <= n_words:
                # If it does, include it in the result
                result.append(element)
            else:
                # If it has more than n_words, print a message
                print(f"{self.STR_PREFIX} Discarded long value: {element}")

        return result
    
    def filter_redundant_substrings(self, response: List[str]) -> List[str]:
        print(f"{self.STR_PREFIX} Removing redundant substrings...", flush=True)
        
        # Set to store values that are substrings of other values
        redundant_elements = set()
        
        # Compare each value with every other value
        for element_i in response:
            for element_j in response:
                if element_i != element_j and element_i in element_j:
                    print(f"{self.STR_PREFIX} Discarded redundant substring: {element_j} (contains '{element_i}')")
                    redundant_elements.add(element_j)
        
        # Build the result dictionary, excluding redundant values
        return [value for value in response if value not in redundant_elements]

    def filter_duplicate_plurals(self, response: List[str]) -> List[str]:
        print(f"{self.STR_PREFIX} Removing duplicate plural words...", flush=True)

        # Step 1: Collect all individual words from all elements
        all_words = set()
        for element in response:
            all_words.update(element.split())

        unique_elements = set()

        # Step 2: Process each element in the list
        for element in response:
            words_in_element = element.split()
            filtered_words = []

            for word in words_in_element:
                original_word = word
                keep_word = True

                # Check if removing "es" results in a word that exists in all_words
                if original_word.endswith("es"):
                    singular_es = original_word[:-2]
                    if singular_es in all_words:
                        print(f"{self.STR_PREFIX} Discarded plural word: {original_word}")
                        keep_word = False

                # If "es" was not removed, check if removing "s" results in a word that exists in all_words
                if keep_word and original_word.endswith("s"):
                    singular_s = original_word[:-1]
                    if singular_s in all_words:
                        print(f"{self.STR_PREFIX} Discarded plural word: {original_word}")
                        keep_word = False

                if keep_word:
                    filtered_words.append(original_word)

            # Reconstruct the filtered element
            filtered_element = ' '.join(filtered_words)
            if filtered_element:
                unique_elements.add(filtered_element)

        return list(unique_elements)
    
    def execute_direct_lvlm_tagging(self) -> List[str]:
        """
        Execute the Direct Tagging with LVLM.
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
            final_json = self.response_fusion(correct_json)
        # Otherwise, use the single response
        else:
            final_json = correct_json[0]

        # Convert the final response to lowercase
        final_json = [element.lower() for element in final_json]

        # If self.banned_words has been loaded, remove banned words from the output
        if config.get(TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS) and config.get(TAGGING_DIRECT_LVLM_BANNED_WORDS) is not None:
            final_json = self.filter_banned_words(final_json)

        # Remove values with more than a certain number of words
        final_json = self.filter_long_values(final_json)

        # Remove words that contain a substring that is also in the dictionary
        final_json = self.filter_redundant_substrings(final_json)

        # Remove duplicate plural words
        final_json = self.filter_duplicate_plurals(final_json)

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4))
        return final_json

    # Override
    def execute(self) -> List[str]:
        pass

    @abstractmethod
    def chat_lvlm(self) -> str:
        pass
