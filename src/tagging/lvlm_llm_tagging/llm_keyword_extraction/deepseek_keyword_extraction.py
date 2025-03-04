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
        save_file: bool = True,  # Whether to save the classification results to a file
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initializes the paths, sets the timeout, and creates the classification directory.
        """

        print(f"\n{self.STR_PREFIX} Initializing DeepSeek keyword extractor...", end=" ")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.deepseek_model_name = deepseek_model_name
        self.save_file = save_file
        self.timeout = timeout

        # Load prompt from prompt.txt
        prompt_path = os.path.join(self.script_dir, "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

        if save_file:
            # Output tags directory path
            output_tags_dir = os.path.join(
                self.script_dir,
                "..",
                "..",
                "output_tags"
            )

            # Create the output directory if it does not exist
            os.makedirs(output_tags_dir, exist_ok=True)

            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"tags_deepseek_{timestamp}.json"
            self.output_file = os.path.join(output_tags_dir, output_filename)
        
        print("Done.\n")

    def load_description(self, pipeline_description: str = None) -> None:
        print(f"{self.STR_PREFIX} Loading input description...", end=" ")

        # If pipeline_description is provided, use it
        if pipeline_description is not None:
            self.input_description = pipeline_description

        # Otherwise, read the most recent .txt file from output_descriptions
        else:
            # Input descriptions directory path
            descriptions_dir = os.path.join(
                self.script_dir,
                "..",
                "lvlm_description",
                "output_descriptions"
            )

            # Gather all .txt files in descriptions_dir
            txt_files = [
                os.path.join(descriptions_dir, f)
                for f in os.listdir(descriptions_dir)
                if f.endswith(".txt")
            ]
            if not txt_files:
                raise FileNotFoundError(f"\n{self.STR_PREFIX} No .txt files found in {descriptions_dir}")

            # Select and read the most recently modified .txt file
            latest_txt_path = max(txt_files, key=os.path.getmtime)
            
            with open(latest_txt_path, "r", encoding="utf-8") as f:
                self.input_description = f.read()

        print("Done.\n")
        
    def remove_thoughts(self, text: str) -> str:
        """
        Removes the <think></think> part of the answer.
        """
        return text.split("</think>")[1]

    def correct_answer_format(self, text: str) -> dict:
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

    def run(self) -> dict:
        """
        Main workflow # TODO
        """
        print(f"{self.STR_PREFIX} Running DeepSeek keyword extraction...\n")        

        start_time = time.time()
        correct_json = None
        while time.time() - start_time < self.timeout:
            response = ollama.chat(
                model=self.deepseek_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt + "\n" + self.input_description
                    }
                ]
            )

            deepseek_answer = response["message"]["content"]

            print(f"{self.STR_PREFIX} DeepSeek answer:\n\n", deepseek_answer + "\n")

            # Remove the <think></think> part of the answer
            deepseek_answer = self.remove_thoughts(deepseek_answer)            

            # Check if the answer is in the correct format
            correct_json = self.correct_answer_format(deepseek_answer)
            if correct_json is not None:
                break
            else:
                print(f"{self.STR_PREFIX} The answer is not in the correct format. Trying again...\n")
        else:
            raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving a correct answer format.\n")

        if self.save_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(correct_json, f, ensure_ascii=False, indent=4)

            print(f"{self.STR_PREFIX} Deepseek answer substring saved to {self.output_file}\n")

        print(f"{self.STR_PREFIX} Final correct answer:\n\n", json.dumps(correct_json, indent=4) + "\n")
        return correct_json

def main():    
    keyword_extractor = DeepseekKeywordExtractor()
    keyword_extractor.load_description()
    keyword_extractor.run()

if __name__ == "__main__":
    main()