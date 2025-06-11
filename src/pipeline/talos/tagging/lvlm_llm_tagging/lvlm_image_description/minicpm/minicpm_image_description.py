from typing import List
import atexit
import subprocess

import ollama

from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.base_image_description import BaseLvlmImageDescriptor

class MiniCpmImageDescriptor(BaseLvlmImageDescriptor):
    """
    [Tagging -> LVLM + LLM -> LVLM Image Description -> MiniCPM]
    
    LVLM image description implementation for the Tagging stage with the LVLM + LLM method that leverages the MiniCPM model.
    """

    STR_PREFIX = "\n[TAGGING | LVLM IMAGE DESCRIPTION | MINICPM]" # Prefix for logging
    ALIAS = "minicpm"  # Alias for filenames
    ALIAS_UPPER = "MINICPM"  # Alias for logging

    def __init__(
        self,
        minicpm_model_name: str = "minicpm-v:8b"
    ):
        """
        Initialize the MiniCPM image descriptor.
        """
        print(f"{self.STR_PREFIX} Initializing MiniCPM image descriptor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.minicpm_model_name = minicpm_model_name

        # Register the cleanup function to clear the model when the object is deleted
        atexit.register(self.__clear_model)

        print("Done.")
    
    # Override from ITaggingLvlmStrategy -> BaseLvlmImageDescriptor
    def execute(self) -> List[str]:
        """
        Execute the LVLM image description with MiniCPM.
        """
        print(f"{self.STR_PREFIX} Running LVLM image description with MiniCPM...", flush=True)

        descriptions = self.execute_image_description()
        return descriptions

    # Override from BaseLvlmImageDescriptor
    def chat_lvlm(self) -> str:
        """
        Generate output tags for the input image using MiniCPM.

        This method will be called by the superclass.
        """
        response = ollama.chat(
            model=self.minicpm_model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [self.input_image_path]
                }
            ]
        )

        return response["message"]["content"]

    def __clear_model(self):
        """
        Clear the Ollama MiniCPM model from the memory.
        """
        print(f"{self.STR_PREFIX} Stopping MiniCPM model...")
        subprocess.run(["ollama", "stop", self.minicpm_model_name])
        print("Done.")

    
def main():
    """
    Main function to run the Qwen image descriptor.
    """
    image_descriptor = MiniCpmImageDescriptor()

    input_image_name = "avocado.jpeg"
    image_descriptor.load_inputs(input_image_name)

    descriptions = image_descriptor.execute()

    image_descriptor.save_outputs(descriptions)

if __name__ == "__main__":
    main()