from typing import List
import subprocess
import atexit
import ollama

from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.base_image_description import BaseLvlmImageDescriptor

class LlavaImageDescriptor(BaseLvlmImageDescriptor):
    """
    [Tagging -> LVLM + LLM -> LVLM Image Description -> LLaVA]
    
    LVLM image description implementation for the Tagging stage with the LVLM + LLM method that leverages the LLaVA model.
    """

    STR_PREFIX = "\n[TAGGING | LVLM IMAGE DESCRIPTION | LLAVA]" # Prefix for logging
    ALIAS = "llava"  # Alias for filenames
    ALIAS_UPPER = "LLAVA"  # Alias for logging

    def __init__(
        self,
        llava_model_name: str = "llava:34b"
    ):
        """
        Initialize the LLaVA image descriptor.
        """
        print(f"{self.STR_PREFIX} Initializing LLaVA image descriptor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.llava_model_name = llava_model_name

        # Register the cleanup function to clear the model when the object is deleted
        atexit.register(self.clear_model)

        print("Done.")
    
    # Override from ITaggingLvlmStrategy -> BaseLvlmImageDescriptor
    def execute(self) -> List[str]:
        """
        Execute the LVLM image description with LLaVA.
        """
        print(f"{self.STR_PREFIX} Running LVLM image description with LLaVA...", flush=True)

        descriptions = self.execute_image_description()
        return descriptions

    # Override from BaseLvlmImageDescriptor
    def chat_lvlm(self) -> str:
        """
        Generate a description for the input image using LLaVA.

        This method will be called by the superclass.
        """
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

        return response["message"]["content"]
    
    def clear_model(self):
        """
        Clear the Ollama LLaVA model from the memory.
        """
        print(f"{self.STR_PREFIX} Stopping LLaVA model...")
        subprocess.run(["ollama", "stop", self.llava_model_name])
        print("Done.")

    
def main():
    """
    Main function to run the LLaVA image descriptor.
    """
    image_descriptor = LlavaImageDescriptor()

    input_image_name = "avocado.jpeg"
    image_descriptor.load_inputs(input_image_name)

    descriptions = image_descriptor.execute()

    image_descriptor.save_outputs(descriptions)

if __name__ == "__main__":
    main()