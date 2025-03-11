import os
import ollama
import time
import json
import re

class DeepseekKeywordExtractor:
    """
    A class to extract keywords from an image description using the DeepSeek model.
    """

    STR_PREFIX = "[TAGGING | KEYWORD EXTRACTION | DEEPSEEK]"

    def __init__(
        self,
        deepseek_model_name: str = "deepseek-r1:14b",
        enhance_output: bool = False,  # Whether to enhance the output with an additional prompt
        save_file: bool = True,  # Whether to save the classification results to a file
        timeout: int = 200  # Timeout in seconds
    ):
        """
        Initializes the paths, sets the timeout, and creates the classification directory.
        """

        print(f"\n{self.STR_PREFIX} Initializing DeepSeek keyword extractor...", end=" ")

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
        
        print("Done.\n")

    def load_description(self, pipeline_description: list[str] = None) -> None:
        print(f"{self.STR_PREFIX} Loading input description(s)...", end=" ")

        # If pipeline_description is provided, use it
        if pipeline_description is not None:
            self.input_description = pipeline_description
            self.iters = len(pipeline_description)
            print(f"Loaded {self.iters} description(s).\n\n")
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
            raise FileNotFoundError(f"\n{self.STR_PREFIX} No directories found in {descriptions_parent_dir}")

        # Select the most recently modified directory
        latest_dir = max(subdirectories, key=os.path.getmtime)

        # Gather all .txt files inside the selected directory
        txt_files = [
            os.path.join(latest_dir, f)
            for f in os.listdir(latest_dir)
            if f.endswith(".txt")
        ]

        if not txt_files:
            raise FileNotFoundError(f"\n{self.STR_PREFIX} No .txt files found in {latest_dir}")

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
        start_index = text.find('{')
        end_index = text.rfind('}')

        # If we can't find a proper pair of braces, return None
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return None

        # Extract the substring that includes the braces
        substring = text[start_index:end_index + 1]        

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
        print(f"{self.STR_PREFIX} Enhancing the output with an additional prompt...", end=" ")

        prompt = self.prompt2 + "\n\n" + response

        while True:
            # Chat with DeepSeek using the second prompt
            deepseek_response = self.chat_deepseek(prompt=prompt)

            print(f"DeepSeek response:\n\n", deepseek_response + "\n")

            # Check if the response is in the correct format
            correct_json = self.correct_response_format(deepseek_response)

            if correct_json is not None:
                break
            else:
                print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...\n")
        return correct_json

    def response_fusion(self, responses: list[dict]) -> dict:
        """
        Merges the responses of multiple iterations into a single response,
        ensuring unique values with numeric keys.
        """
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

        return final_response
    
    def remove_duplicates(self, response: dict) -> dict:
        unique_values = set()
        result = {}

        for key, value in response.items():
            lower_value = value.lower()
            if lower_value not in unique_values:
                unique_values.add(lower_value)
                result[key] = lower_value

        return result

    def remove_duplicate_plurals(self, response: dict) -> dict:
        values = list(response.values())

        unique_values = {
            word for word in values
            if not (word.endswith('s') and word[:-1] in values)
        }

        return {k: v for k, v in response.items() if v in unique_values}

    def run(self) -> dict:
        """
        Main workflow # TODO
        """
        print(f"{self.STR_PREFIX} Running DeepSeek keyword extraction...\n", flush=True)

        start_time = time.time()
        correct_json = [None] * self.iters

        for i in range(self.iters):
            if self.iters > 1:
                print(f"{self.STR_PREFIX} Iteration {i + 1}/{self.iters}...\n")

            prompt = self.prompt1 + "\n" + self.input_description[i]

            while time.time() - start_time < self.timeout:
                # Chat with DeepSeek using the first prompt
                deepseek_response = self.chat_deepseek(prompt=prompt)

                print(f"DeepSeek response:\n\n", deepseek_response + "\n")

                # Check if the response is in the correct format
                correct_json[i] = self.correct_response_format(deepseek_response)

                if correct_json[i] is not None:
                    # Filter the response to improve results if enhance_output is True and the response has at least one element
                    if self.enhance_output and len(correct_json[i]) > 0:
                        correct_json[i] = self.enhance_response(deepseek_response)
                    break
                else:
                    print(f"{self.STR_PREFIX} The response is not in the correct format. Trying again...\n")
            else:
                raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving a correct response format.\n")

        # Merge the responses if there are multiple iterations
        final_json = self.response_fusion(correct_json) if self.iters > 1 else correct_json[0]

        # Remove duplicate words
        final_json = self.remove_duplicates(final_json)

        # Remove duplicate plural words
        final_json = self.remove_duplicate_plurals(final_json)

        if self.save_file:
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"tags_deepseek_{timestamp}.json"
            output_file = os.path.join(self.output_tags_dir, output_filename)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=4)

            print(f"{self.STR_PREFIX} DeepSeek response substring saved to: {output_file}\n")

        print(f"{self.STR_PREFIX} Final response substring:\n\n", json.dumps(final_json, indent=4) + "\n")
        return final_json

def main():    
    keyword_extractor = DeepseekKeywordExtractor()
    keyword_extractor.load_description()
    keyword_extractor.run()

if __name__ == "__main__":
    main()