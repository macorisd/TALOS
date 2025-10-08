from typing import List
import os
import atexit
import subprocess
import tempfile
from pathlib import Path

import ollama

from pipeline.config.paths import (
    INPUT_IMAGES_DIR
)

from pipeline.talos.tagging.direct_lvlm_tagging.base_direct_lvlm_tagging import BaseDirectLvlmTagger

class MiniCpmTagger(BaseDirectLvlmTagger):
    """
    [Tagging -> Direct Tagging with LVLM -> MiniCPM]
    
    Tagging stage implementation that leverages the MiniCPM model.
    """

    STR_PREFIX = "\n[TAGGING | MINICPM]" # Prefix for logging
    ALIAS = "minicpm" # Alias for filenames

    def __init__(
        self,
        minicpm_model_name: str = "minicpm-v:8b"
    ):
        """
        Initialize the MiniCPM tagger.
        """
        print(f"{self.STR_PREFIX} Initializing MiniCPM tagger...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Variables
        self.minicpm_model_name = minicpm_model_name

        # Register the cleanup function to clear the model when the object is deleted
        atexit.register(self.__clear_model)

        print("Done.")
    
    # Override from ITaggingStrategy -> BaseTagger
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image's path.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ")
        self.input_image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        if not os.path.isfile(self.input_image_path):
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {self.input_image_name} was not found.")
        print("Done.")
    
    # Override from ITaggingStrategy -> BaseTagger -> BaseDirectLvlmTagger
    def execute(self) -> List[str]:
        """
        Execute the MiniCPM Tagging.
        """
        print(f"{self.STR_PREFIX} Running Tagging with MiniCPM...", flush=True)

        tags = self.execute_direct_lvlm_tagging()
        return tags

    # Override from BaseDirectLvlmTagger
    def chat_lvlm(self) -> str:
        """
        Generate output tags for the input image using MiniCPM.

        This method will be called by the superclass.
        """

        if getattr(self, "input_image_path", None) is None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                self.input_image.save(tmp, format="PNG")
                self.input_image_path = Path(tmp.name)

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
    Main function to run the Tagging stage with MiniCPM.
    """
    tagger = MiniCpmTagger()
    
    input_image_name = "avocado.jpeg"
    tagger.load_inputs(input_image_name)
    tags = tagger.execute()
    
    tagger.save_outputs(tags)

if __name__ == "__main__":
    main()