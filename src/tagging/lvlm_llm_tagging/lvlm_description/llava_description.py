import ollama

import os
import time

class LlavaDescriptor:
    """
    A class to describe an image using Ollama's LLaVA model.
    """

    STR_PREFIX = "[TAGGING | DESCRIPTION | LLAVA]"

    def __init__(
        self,        
        llava_model_name: str = "llava:34b",        
        prompt: str = "Describe the image.",
        save_file: bool = True,  # Whether to save the description results to a file
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """

        print(f"\n{self.STR_PREFIX} Initializing LLaVA image descriptor...", end=" ")        

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.llava_model_name = llava_model_name        
        self.prompt = prompt
        self.save_file = save_file
        self.timeout = timeout
        
        if save_file:
            # Output descriptions directory path
            output_descriptions_dir = os.path.join(
                self.script_dir, 
                "output_descriptions"
            )

            # Create the output directory if it does not exist
            os.makedirs(output_descriptions_dir, exist_ok=True)

            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"description_llava_{timestamp}.txt"
            self.output_file = os.path.join(output_descriptions_dir, output_filename)
        
        print("Done.\n")

    def load_image_path(self, input_image_name: str) -> None:
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ")

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
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {self.input_image_name} was not found.\n")
        
        print("Done.\n")

    def run(self) -> str:
        """
        Generates a description for the loaded image using Ollama's LLaVA model,
        and writes the description to a text file.
        """
        print(f"{self.STR_PREFIX} Running LLaVA image descriptor...", end=" ")
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
                print(f"{self.STR_PREFIX} The description is empty. Trying again...\n")
        else:
            raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving a valid description.\n")

        # Print the description                   
        print(f"Image description:\n\n" + description + "\n")

        # Save the description to a text file if saving is enabled
        if self.save_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(description)
            print(f"{self.STR_PREFIX} Description saved to: {self.output_file}\n")
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Description was not saved.\n")

        return description


def main():
    descriptor = LlavaDescriptor()
    descriptor.load_image_path("desk.jpg")
    descriptor.run()


if __name__ == "__main__":
    main()
