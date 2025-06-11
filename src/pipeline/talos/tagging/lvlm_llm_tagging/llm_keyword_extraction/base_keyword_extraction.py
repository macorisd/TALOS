from abc import abstractmethod
import json
import os
import time
from typing import List

from pipeline.strategy.strategy import ITaggingLlmStrategy
from pipeline.common.large_model_tagging import LargeModelForTagging
from pipeline.config.paths import (
    OUTPUT_DESCRIPTIONS_DIR,
    OUTPUT_TAGS_DIR,
    TAGGING_LLM_PROMPT1,
    TAGGING_LLM_PROMPT2
)
from pipeline.config.config import (
    config,
    SAVE_INTERMEDIATE_FILES,
    TAGGING_LLM_ENHANCE_OUTPUT,
    TAGGING_LLM_EXCLUDE_BANNED_WORDS,
    TAGGING_LLM_BANNED_WORDS,
    TAGGING_LLM_TIMEOUT
)

class BaseLlmKeywordExtractor(ITaggingLlmStrategy, LargeModelForTagging):
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
        
        if SAVE_INTERMEDIATE_FILES:
            # Create output directory if it does not exist
            os.makedirs(OUTPUT_TAGS_DIR, exist_ok=True)

    # Override from ITaggingLlmStrategy
    def load_inputs(self, descriptions: List[str] = None) -> None:
        """
        Load the LLM Keyword Extraction inputs.
        """
        self.load_descriptions(descriptions)

    # Override from ITaggingLlmStrategy
    def load_descriptions(self, descriptions: List[str] = None) -> None:
        """
        Load the input descriptions for keyword extraction.

        If 'descriptions' is provided, use it.
        Otherwise, load the most recent descriptions from the LVLM image description's output directory.
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

    @abstractmethod # from ITaggingLlmStrategy
    def execute(self) -> List[str]:
        raise NotImplementedError("execute method must be implemented in subclasses.")

    def execute_keyword_extraction(self) -> List[str]:
        """
        Execute the LLM Keyword Extraction for the Tagging with LVLM + LLM.

        This method iteratively prompts the LLM model to extract keywords from the input descriptions.

        This method will be called by the execute method in the subclasses.
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
            print(f"{self.STR_PREFIX} {self.iters} iterations completed. Merging the responses...", end=" ", flush=True)
            final_json = self.response_fusion(correct_json)
        # Otherwise, use the single response
        else:
            final_json = correct_json[0]

        # Convert the final response to lowercase
        final_json = [element.lower() for element in final_json]

        # If self.banned_words has been loaded, remove banned words from the output
        if config.get(TAGGING_LLM_EXCLUDE_BANNED_WORDS) and config.get(TAGGING_LLM_BANNED_WORDS) is not None:
            final_json = self.filter_banned_words(
                response=final_json, 
                banned_words=config.get(TAGGING_LLM_BANNED_WORDS)
            )

        # Remove values with more than a certain number of words
        final_json = self.filter_long_values(final_json)

        # Remove words that contain a substring that is also in the dictionary
        final_json = self.filter_redundant_substrings(final_json)

        # Remove duplicate plural words
        final_json = self.filter_duplicate_plurals(final_json)

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4))
        return final_json
    
    # Override from ITaggingLlmStrategy
    def save_outputs(self, tags: List[str]) -> None:
        """
        Save the generated keywords to text files.
        """
        if config.get(SAVE_INTERMEDIATE_FILES):
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_tags(tags, timestamp)
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Keyword extraction output was not saved.")
    
    # Override from ITaggingLlmStrategy
    def save_tags(self, tags: List[str], timestamp: str) -> None:
        """
        Save the generated keywords to text files.
        """
        if SAVE_INTERMEDIATE_FILES:
            output_filename = f"tags_{timestamp}_{self.ALIAS}.json"
            output_file = os.path.join(OUTPUT_TAGS_DIR, output_filename)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tags, f, ensure_ascii=False, indent=4)
            print(f"{self.STR_PREFIX} Tags saved to: {output_file}")
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Keyword extraction output was not saved.")
    
    @abstractmethod
    def chat_llm(self, prompt: str) -> str:
        raise NotImplementedError("chat_llm method must be implemented in subclasses.")
