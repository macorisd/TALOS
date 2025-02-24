import ollama

import os
from PIL import Image
import time

class LlavaDescriptor:
    """
    A class to describe an image using Ollama's LLaVA model.
    """

    STR_PREFIX = "[DESCRIPTION | LLAVA]"

    def __init__(
        self,        
        llava_model_name: str = "llava:34b",
        input_image_name: str = "input_image.png",
        prompt: str = "Describe the image.",
        save_file: bool = True,  # Whether to save the description results to a file
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"\n{self.STR_PREFIX} Initializing LLaVA descriptor...\n")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.llava_model_name = llava_model_name        
        self.prompt = prompt
        self.save_file = save_file
        self.timeout = timeout

        # Input image path
        self.input_image_path = os.path.join(
            self.script_dir, 
            "..", 
            "..",
            "..",
            "input_images",
            input_image_name
        )
        
        # Check if the image exists
        if not os.path.isfile(self.input_image_path):
            raise FileNotFoundError(f"\n{self.STR_PREFIX} The image {self.input_image_name} was not found.")

        # Output descriptions directory path
        self.descriptions_dir = os.path.join(
            self.script_dir, 
            "output_descriptions"
        )

        # Create the output directory if it does not exist
        os.makedirs(self.descriptions_dir, exist_ok=True)

        # Prepare timestamped output file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_filename = f"description_{timestamp}.txt"
        self.output_file = os.path.join(self.descriptions_dir, self.output_filename)        

    def describe_image(self) -> str:
        """
        Generates a description for the loaded image using Ollama's LLaVA model,
        and writes the description to a text file.
        """
        description = ""
        start_time = time.time()  # Start timer for timeout
        
        # Describe the image
        while time.time() - start_time < self.timeout:                
            response = ollama.chat(
                model=self.llava_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [self.input_image_path]
                    }
                ]
            )
            description = response["message"]["content"]
            if description.strip(): # Not empty
                break
            else:
                print(f"\n{self.STR_PREFIX} The description is empty. Trying again...\n")
        else:
            raise TimeoutError(f"\n{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving a valid description.")

        # Print the description                   
        print(f"\n{self.STR_PREFIX} Image description: " + description + "\n")

        # Save the description to a text file if saving is enabled
        if self.save_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(description)
            print(f"\n{self.STR_PREFIX} Description saved in {self.output_file}")
        else:
            print(f"\n{self.STR_PREFIX} Saving file is disabled. Description was not saved.")

        return description


def main():
    descriptor = LlavaDescriptor(input_image_name="desk.jpg")    
    descriptor.describe_image()


if __name__ == "__main__":
    main()
