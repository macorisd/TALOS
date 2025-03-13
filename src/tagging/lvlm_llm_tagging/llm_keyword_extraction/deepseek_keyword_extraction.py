import os
import ollama
import time
import json
import re

class DeepseekKeywordExtractor:
    """
    A class to extract keywords from an image description using the DeepSeek model.
    """

    STR_PREFIX = "\n[TAGGING | KEYWORD EXTRACTION | DEEPSEEK]"

    def __init__(
        self,
        deepseek_model_name: str = "deepseek-r1:14b",
        exclude_banned_words: bool = False,  # Whether to exclude banned words from the output
        enhance_output: bool = False,  # Whether to enhance the output with an additional prompt
        save_file: bool = True,  # Whether to save the classification results to a file
        timeout: int = 200  # Timeout in seconds
    ):
        """
        Initializes the paths, sets the timeout, and creates the classification directory.
        """

        print(f"{self.STR_PREFIX} Initializing DeepSeek keyword extractor...", end=" ", flush=True)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.deepseek_model_name = deepseek_model_name
        self.enhance_output = enhance_output
        self.save_file = save_file
        self.timeout = timeout if timeout > 0 else 200

        self.input_description = []
        self.iters = 1

        # Load first prompt from prompt1.txt
        prompt1_path = os.path.join(self.script_dir, "prompts", "prompt1.txt")
        with open(prompt1_path, "r", encoding="utf-8") as f:
            self.prompt1 = f.read()

        # If enhance_output is True, load the second prompt from prompt2.txt
        if enhance_output:
            prompt2_path = os.path.join(self.script_dir, "prompts", "prompt2.txt")
            with open(prompt2_path, "r", encoding="utf-8") as f:
                self.prompt2 = f.read()

        # If remove_banned_words is True, load the banned words from banned_words.json
        if exclude_banned_words:
            banned_words_path = os.path.join(
                self.script_dir,
                "..",
                "..",
                "banned_words.json"
            )

            with open(banned_words_path, "r", encoding="utf-8") as f:
                self.banned_words = json.load(f)
        else:
            self.banned_words = None

        if save_file:
            # Output tags directory path
            self.output_tags_dir = os.path.join(
                self.script_dir,
                "..",
                "..",
                "output_tags"
            )

            # Create the output directory if it does not exist
            os.makedirs(self.output_tags_dir, exist_ok=True)
        
        print("Done.")

    def load_description(self, pipeline_description: list[str] = None) -> None:
        print(f"{self.STR_PREFIX} Loading input description(s)...", end=" ", flush=True)

        # If pipeline_description is provided, use it
        if pipeline_description is not None:
            self.input_description = pipeline_description
            self.iters = len(pipeline_description)
            print(f"Loaded {self.iters} description(s).")
            return

        # Input descriptions parent directory
        descriptions_parent_dir = os.path.join(
            self.script_dir,
            "..",
            "lvlm_description",
            "output_descriptions"
        )

        # Get all subdirectories inside output_descriptions
        subdirectories = [
            os.path.join(descriptions_parent_dir, d)
            for d in os.listdir(descriptions_parent_dir)
            if os.path.isdir(os.path.join(descriptions_parent_dir, d))
        ]

        if not subdirectories:
            raise FileNotFoundError(f"{self.STR_PREFIX} No directories found in {descriptions_parent_dir}")

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


    def remove_thoughts(self, text: str) -> str:
        """
        Removes the <think></think> part of the response.
        """
        return text.split("</think>")[1]

    def chat_deepseek(self, prompt: str) -> str:
        response = ollama.chat(
                model=self.deepseek_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        
        response_content = response["message"]["content"]

        # Remove the <think></think> part of the response
        response_content = self.remove_thoughts(response_content)

        return response_content.strip()

    def correct_response_format(self, text: str) -> dict:
        """
        Checks if there is a substring within the given text that starts with '{' and ends with '}'
        and follows the exact format:

        {
            "1": "[phrase]",
            "2": "[phrase]",
            ...
            "n": "[phrase]"
        }

        If the format is correct, returns the JSON substring. Otherwise, returns None.
        """
        # Find indices of the first '{' and the last '}'
        start_index = text.rfind('{')
        end_index = text.rfind('}')

        # If we can't find a proper pair of braces, return None
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return None

        # Extract the substring that includes the braces
        substring = text[start_index:end_index + 1]

        # If there's a "/" character in the substring, it's not a valid JSON
        if "/" in substring:
            return None

        # Delete all "\" characters from the substring
        substring = substring.replace("\\", "")

        # Attempt to parse the substring as JSON
        try:
            parsed_data = json.loads(substring)
        except json.JSONDecodeError:
            return None

        # The parsed data must be a dictionary with at least one item
        if not isinstance(parsed_data, dict):
            return None

        # Check each key-value pair in the dictionary
        for key, value in parsed_data.items():
            # Key must be a string representing a number
            if not re.match(r"^\d+$", key):
                return None

            # Value must be a string
            if not isinstance(value, str):
                return None

        return parsed_data
    
    def enhance_response(self, response: str) -> dict:
        """
        Additional DeepSeek prompt to enhance the keyword extraction.
        """
        print(f"{self.STR_PREFIX} Enhancing the output with an additional prompt...", end=" ", flush=True)

        prompt = self.prompt2 + "\n\n" + response

        while True:
            # Chat with DeepSeek using the second prompt
            deepseek_response = self.chat_deepseek(prompt=prompt)

            print(f"{self.STR_PREFIX} DeepSeek response:\n\n", deepseek_response)

            # Check if the response is in the correct format
            correct_json = self.correct_response_format(deepseek_response)

            if correct_json is not None:
                break
            else:
                print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...")
        return correct_json

    def response_fusion(self, responses: list[dict]) -> dict:
        """
        Merges the responses of multiple iterations into a single response,
        ensuring unique values with numeric keys.
        """
        print(f"{self.STR_PREFIX} {self.iters} iterations completed. Merging the responses...", end=" ", flush=True)

        final_response = {}
        unique_values = set()
        index = 1

        # Collect all unique values from the dictionaries
        for response in responses:
            for value in response.values():
                if value not in unique_values:
                    unique_values.add(value)
                    final_response[str(index)] = value
                    index += 1

        print("Done.")

        print(f"{self.STR_PREFIX} Merged response substring:\n\n", json.dumps(final_response, indent=4))

        return final_response
    
    def remove_banned_words(self, response: dict) -> dict:
        print(f"{self.STR_PREFIX} Removing banned words...", flush=True)

        banned_words = set(self.banned_words)

        for word in banned_words:
            for _, value in response.items():
                if word in value:
                    print(f"{self.STR_PREFIX} Discarded banned word: {word}")
                    response = {k: v for k, v in response.items() if word not in v}
                    break

        return response
    
    def remove_duplicates(self, response: dict) -> dict:
        print(f"{self.STR_PREFIX} Removing duplicate words...", flush=True)

        # Set to store unique values
        unique_values = set()
        result = {}
        
        for key, value in response.items():
            # Check if the value is already in the unique_values set
            if value not in unique_values:
                # If not, add it to the set and include it in the result
                unique_values.add(value)
                result[key] = value
            else:
                # If it is a duplicate, print a message
                print(f"{self.STR_PREFIX} Discarded duplicate word: {value}")

        return result
    
    def remove_redundant_substrings(self, response: dict) -> dict:
        print(f"{self.STR_PREFIX} Removing redundant substrings...", flush=True)
        
        values = list(response.values())
        
        # Set to store values that are substrings of other values
        redundant_values = set()
        
        # Compare each value with every other value
        for i in range(len(values)):
            for j in range(len(values)):
                if i != j and values[i] in values[j]:
                    print(f"{self.STR_PREFIX} Discarded redundant substring: {values[j]} (contains '{values[i]}')")
                    redundant_values.add(values[j])
        
        # Build the result dictionary, excluding redundant values
        return {k: v for k, v in response.items() if v not in redundant_values}

    def remove_duplicate_plurals(self, response: dict) -> dict:
        print(f"{self.STR_PREFIX} Removing duplicate plural words...", flush=True)

        # Step 1: Collect all individual words from all values
        all_words = set()
        for value in response.values():
            all_words.update(value.split())
        
        unique_values = set()
        
        # Step 2: Process each value in the dictionary
        for value in response.values():
            words_in_value = value.split()
            filtered_words = []
            
            for word in words_in_value:
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
            
            # Reconstruct the filtered value
            filtered_value = ' '.join(filtered_words)
            if filtered_value:
                unique_values.add(filtered_value)
        
        # Keep only the entries whose original value is in unique_values
        return {k: v for k, v in response.items() if v in unique_values}

    def run(self) -> dict:
        """
        Main workflow # TODO
        """
        print(f"{self.STR_PREFIX} Running DeepSeek keyword extraction...", flush=True)

        start_time = time.time()
        correct_json = [None] * self.iters

        for i in range(self.iters):
            if self.iters > 1:
                print(f"{self.STR_PREFIX} Iteration {i + 1}/{self.iters}...")

            prompt = self.prompt1 + "\n" + self.input_description[i]

            while time.time() - start_time < self.timeout:
                # Chat with DeepSeek using the first prompt
                deepseek_response = self.chat_deepseek(prompt=prompt)

                print(f"{self.STR_PREFIX} DeepSeek response:\n\n", deepseek_response + "\n")

                # Check if the response is in the correct format
                correct_json[i] = self.correct_response_format(deepseek_response)

                if correct_json[i] is not None:
                    # Filter the response to improve results if enhance_output is True and the response has at least one element
                    if self.enhance_output and len(correct_json[i]) > 0:
                        correct_json[i] = self.enhance_response(deepseek_response)
                    break
                else:
                    print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...")
            else:
                raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving a correct response format.")

        # Merge the responses if there are multiple iterations
        if self.iters > 1:
            final_json = self.response_fusion(correct_json)
        # Otherwise, use the single response
        else:
            final_json = correct_json[0]

        # Convert the final response to lowercase
        final_json = {k: v.lower() for k, v in final_json.items()}

        # If self.banned_words has been loaded, remove banned words from the output
        if self.banned_words is not None:
            final_json = self.remove_banned_words(final_json)

        # Remove duplicate words
        final_json = self.remove_duplicates(final_json)

        # Remove words that contain a substring that is also in the dictionary
        final_json = self.remove_redundant_substrings(final_json)

        # Remove duplicate plural words
        final_json = self.remove_duplicate_plurals(final_json)

        if self.save_file:
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"tags_deepseek_{timestamp}.json"
            output_file = os.path.join(self.output_tags_dir, output_filename)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=4)

            print(f"{self.STR_PREFIX} DeepSeek response substring saved to: {output_file}")

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4))
        return final_json

def main():    
    keyword_extractor = DeepseekKeywordExtractor(exclude_banned_words=True)
    keyword_extractor.load_description()
    keyword_extractor.run()

if __name__ == "__main__":
    main()