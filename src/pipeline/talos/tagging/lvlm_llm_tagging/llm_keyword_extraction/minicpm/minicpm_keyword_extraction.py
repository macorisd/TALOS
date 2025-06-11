from typing import List
import subprocess
import atexit
import ollama

from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.base_keyword_extraction import BaseLlmKeywordExtractor

class MiniCpmKeywordExtractor(BaseLlmKeywordExtractor):
    """
    [Tagging -> LVLM + LLM -> LLM Keyword Extraction -> MiniCPM]
    
    LLM keyword extraction implementation for the Tagging stage with the LVLM + LLM method that leverages the MiniCPM model.
    """
    STR_PREFIX = "\n[TAGGING | LLM KEYWORD EXTRACTION | MINICPM]"  # Prefix for logging
    ALIAS = "minicpm"  # Alias for filenames
    ALIAS_UPPER = "MINICPM"  # Alias for logging

    def __init__(
        self,
        minicpm_model_name: str = "minicpm-v:8b"
    ):
        """
        Initialize the MiniCPM keyword extractor.
        """
        print(f"{self.STR_PREFIX} Initializing MiniCPM keyword extractor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.minicpm_model_name = minicpm_model_name

        # Register the cleanup function to clear the model when the object is deleted
        atexit.register(self.__clear_model)

        print("Done.")

    # Override from ITaggingLlmStrategy -> BaseLlmKeywordExtractor
    def execute(self) -> List[str]:
        """
        Execute the LLM keyword extraction with MiniCPM.
        """
        print(f"{self.STR_PREFIX} Running LLM keyword extraction with MiniCPM...", flush=True)

        tags = self.execute_keyword_extraction()
        return tags
    
    # Override from BaseLlmKeywordExtractor
    def chat_llm(self, prompt: str) -> str:
        """
        Prompt the MiniCPM model with the input text prompt.

        This method will be called by the superclass.
        """
        response = ollama.chat(
                model=self.minicpm_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        
        response_content = response["message"]["content"]

        return response_content.strip()
    
    def __clear_model(self):
        """
        Clear the Ollama MiniCPM model from the memory.
        """
        print(f"{self.STR_PREFIX} Stopping MiniCPM model...")
        subprocess.run(["ollama", "stop", self.minicpm_model_name])
        print("Done.")
    

def main():
    """
    Main function to run the MiniCPM keyword extractor.
    """
    keyword_extractor = MiniCpmKeywordExtractor()

    keyword_extractor.load_inputs()

    tags = keyword_extractor.execute()

    keyword_extractor.save_outputs(tags)


if __name__ == "__main__":
    main()