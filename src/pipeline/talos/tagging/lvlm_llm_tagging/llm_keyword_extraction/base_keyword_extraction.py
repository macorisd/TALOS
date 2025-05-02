from abc import abstractmethod
import json
import os
import time
from typing import List

from strategy.strategy import ITaggingLlmStrategy
from config.paths import (
    OUTPUT_DESCRIPTIONS_DIR,
    OUTPUT_TAGS_DIR,
    TAGGING_LLM_PROMPT1,
    TAGGING_LLM_PROMPT2
)
from config.config import (
    config,
    SAVE_FILES,
    TAGGING_LLM_ENHANCE_OUTPUT,
    TAGGING_LLM_EXCLUDE_BANNED_WORDS,
    TAGGING_LLM_BANNED_WORDS,
    TAGGING_LLM_TIMEOUT
)

class BaseLlmKeywordExtractor(ITaggingLlmStrategy):
    """
    [Tagging -> LVLM + LLM -> LLM Keyword Extraction]
    
    Base class for LLM keyword extractors for the Tagging stage with the LVLM + LLM method.
    """
    def __init__(self):
        """
        Initialize the base LLM keyword extractor.
        """
        # Variables
        with open(TAGGING_LLM_PROMPT1, "r", encoding="utf-8") as f:
            self.prompt1 = f.read()

        # If enhance_output is True, load the second prompt from prompt2.txt
        if config.get(TAGGING_LLM_ENHANCE_OUTPUT):
            with open(TAGGING_LLM_PROMPT2, "r", encoding="utf-8") as f:
                self.prompt2 = f.read()
        
        if SAVE_FILES:
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_TAGS_DIR, exist_ok=True)

    def load_inputs(self, descriptions: List[str] = None) -> None:
        """
        Load the LLM Keyword Extraction inputs.
        """
        self.load_descriptions(descriptions)

    def load_descriptions(self, descriptions: List[str] = None) -> None:
        """
        Load the input descriptions for keyword extraction.

        If 'descriptions' is provided, use it.
        Otherwise, load the most recent descriptions from the LVLM output descriptions directory.
        """
        print(f"{self.STR_PREFIX} Loading input description(s)...", end=" ", flush=True)

        # If descriptions is provided by the pipeline, use it
        if descriptions is not None:
            self.input_description = descriptions
            self.iters = len(descriptions)
            print(f"Loaded {self.iters} description(s).")
            return

        # Get all subdirectories inside the LVLM output descriptions directory
        subdirectories = [
            os.path.join(OUTPUT_DESCRIPTIONS_DIR, d)
            for d in os.listdir(OUTPUT_DESCRIPTIONS_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DESCRIPTIONS_DIR, d))
        ]

        if not subdirectories:
            raise FileNotFoundError(f"{self.STR_PREFIX} No directories found in {OUTPUT_DESCRIPTIONS_DIR}")

        # Select the most recently modified directory
        latest_dir = max(subdirectories, key=os.path.getmtime)

        # Gather all .txt files inside the selected directory
        txt_files = [
            os.path.join(latest_dir, f)
            for f in os.listdir(latest_dir)
            if f.endswith(".txt")
        ]

        if not txt_files:
            raise FileNotFoundError(f"{self.STR_PREFIX} No .txt files found in {latest_dir}")

        # Read all .txt files and store their content in a list
        self.input_description = []
        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8") as f:
                self.input_description.append(f.read())

        self.iters = len(self.input_description)
        print(f"Loaded {self.iters} descriptions.")


    @abstractmethod
    def execute(self) -> List[str]:
        pass


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

    
    def enhance_response(self, response: str) -> dict:
        """
        Additional LLM prompt to enhance the keyword extraction.
        """
        print(f"{self.STR_PREFIX} Enhancing the output with an additional prompt...", end=" ", flush=True)

        prompt = self.prompt2 + "\n\n" + response

        while True:
            # Chat with the LLM using the second prompt
            response = self.chat_llm(prompt=prompt)

            print(f"{self.STR_PREFIX} LLM response:\n\n", response)

            # Check if the response is in the correct format
            correct_json = self.correct_response_format(response)

            if correct_json is not None:
                break
            else:
                print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...")
        return correct_json

    def response_fusion(self, responses: List[List[str]]) -> List[str]:
        """
        Merges the responses of multiple iterations into a single list,
        ensuring unique string values in order.
        """
        print(f"{self.STR_PREFIX} {self.iters} iterations completed. Merging the responses...", end=" ", flush=True)

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

        banned_words = set(config.get(TAGGING_LLM_BANNED_WORDS))

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


    def execute_keyword_extraction(self) -> List[str]:
        """
        Iteratively prompt the LLM model to extract keywords from the input descriptions.
        """
        start_time = time.time()
        correct_json = [None] * self.iters

        for i in range(self.iters):
            if self.iters > 1:
                print(f"{self.STR_PREFIX} Iteration {i + 1}/{self.iters}...")

            prompt = self.prompt1 + "\n" + self.input_description[i]

            while time.time() - start_time < config.get(TAGGING_LLM_TIMEOUT):
                # Chat with the LLM using the first prompt
                response = self.chat_llm(prompt=prompt)

                print(f"{self.STR_PREFIX} LLM response:\n\n", response + "\n")

                # Check if the response is in the correct format
                correct_json[i] = self.correct_response_format(response)

                if correct_json[i] is not None:
                    # Filter the response to improve results if enhance_output is True and the response has at least one element
                    if config.get(TAGGING_LLM_ENHANCE_OUTPUT) and len(correct_json[i]) > 0:
                        correct_json[i] = self.enhance_response(response)
                    break
                else:
                    print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...")
            else:
                raise TimeoutError(f"{self.STR_PREFIX} Timeout of {config.get(TAGGING_LLM_TIMEOUT)} seconds reached without receiving a correct response format.")

        # Merge the responses if there are multiple iterations
        if self.iters > 1:
            final_json = self.response_fusion(correct_json)
        # Otherwise, use the single response
        else:
            final_json = correct_json[0]

        # Convert the final response to lowercase
        final_json = [element.lower() for element in final_json]

        # If self.banned_words has been loaded, remove banned words from the output
        if config.get(TAGGING_LLM_EXCLUDE_BANNED_WORDS) and config.get(TAGGING_LLM_BANNED_WORDS) is not None:
            final_json = self.filter_banned_words(final_json)

        # Remove values with more than a certain number of words
        final_json = self.filter_long_values(final_json)

        # Remove words that contain a substring that is also in the dictionary
        final_json = self.filter_redundant_substrings(final_json)

        # Remove duplicate plural words
        final_json = self.filter_duplicate_plurals(final_json)

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4))
        return final_json
    
    def save_outputs(self, tags: List[str]) -> None:
        """
        Save the generated keywords to text files.
        """
        if SAVE_FILES:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_tags(tags, timestamp)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Keyword extraction output was not saved.")
    
    def save_tags(self, tags: List[str], timestamp: str) -> None:
        """
        Save the generated keywords to text files.
        """
        if SAVE_FILES:
            output_filename = f"tags_{self.ALIAS}_{timestamp}.json"
            output_file = os.path.join(OUTPUT_TAGS_DIR, output_filename)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tags, f, ensure_ascii=False, indent=4)
            print(f"{self.STR_PREFIX} Tags saved to: {output_file}")
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Keyword extraction output was not saved.")
    

    @abstractmethod
    def chat_llm(self, prompt: str) -> str:
        pass
