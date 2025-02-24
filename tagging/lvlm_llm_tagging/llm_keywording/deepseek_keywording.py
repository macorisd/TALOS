import os
import ollama
import time
import json
import re

class DeepseekKeywordExtractor:
    """
    A class to # TODO
    """

    def __init__(
        self,        
        deepseek_model_name: str = "deepseek-r1:14b",
        pipeline_description: str = None,  # The pipeline descriptions to be classified
        save_file: bool = True,  # Whether to save the classification results to a file
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initializes the paths, sets the timeout, and creates the classification directory.
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.deepseek_model_name = deepseek_model_name
        self.pipeline_description = pipeline_description
        self.save_file = save_file
        self.timeout = timeout

        # Load prompt from prompt.txt
        prompt_path = os.path.join(self.script_dir, "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

        if not pipeline_description:
            # .txt file containing the image description (output from the LVLM description)
            self.descriptions_dir = os.path.join(
                self.script_dir,
                "..",
                "lvlm_description",
                "output_descriptions"
            )

        # Directory to store the classification results
        self.keywords_dir = os.path.join(self.script_dir, "output_keywords")
        os.makedirs(self.keywords_dir, exist_ok=True)

    def read_description_from_file(self) -> str:
        """
        Reads the description from the most recent .txt file in the descriptions directory.
        """
        # Gather all .txt files in descriptions_dir
        txt_files = [
            os.path.join(self.descriptions_dir, f)
            for f in os.listdir(self.descriptions_dir)
            if f.endswith(".txt")
        ]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.descriptions_dir}")

        # Select and read the most recently modified .txt file
        latest_txt_path = max(txt_files, key=os.path.getmtime)
        
        with open(latest_txt_path, "r", encoding="utf-8") as f:
            description_content = f.read()

        return description_content

    def correct_answer_format(self, text: str) -> str:
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

        return substring

    def classify(self) -> str:
        """
        Main workflow to locate the most recent .txt description, # TODO
        """
        description_content = self.pipeline_description if self.pipeline_description else self.read_description_from_file()

        start_time = time.time()
        correct_substring = None
        while time.time() - start_time < self.timeout:
            response = ollama.chat(
                model=self.deepseek_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt + "\n" + description_content
                    }
                ]
            )

            deepseek_answer = response["message"]["content"]

            print("Content:\n", deepseek_answer)

            correct_substring = self.correct_answer_format(deepseek_answer)            
            if correct_substring is not None:
                break
            else:
                print("\nThe answer is not in the correct format. Trying again...\n")
        else:
            raise TimeoutError(f"Timeout of {self.timeout} seconds reached without receiving a correct answer format.")

        if self.save_file:
            # Save the Deepseek answer with a timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"keywords_{timestamp}.txt"
            output_path = os.path.join(self.keywords_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(correct_substring)

            print(f"Deepseek answer saved to {output_path}")

        return correct_substring

def main():    
    classifier = DeepseekKeywordExtractor(save_file=True)
    final_answer = classifier.classify()
    print("\nFinal correct answer substring:\n", final_answer)

if __name__ == "__main__":
    main()