from typing import List
import subprocess
import atexit
import ollama

from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.base_keyword_extraction import BaseLlmKeywordExtractor

class DeepseekKeywordExtractor(BaseLlmKeywordExtractor):
    """
    [Tagging -> LVLM + LLM -> LLM Keyword Extraction -> DeepSeek]
    
    LLM keyword extraction implementation for the Tagging stage with the LVLM + LLM method that leverages the DeepSeek model.
    """
    STR_PREFIX = "\n[TAGGING | LLM KEYWORD EXTRACTION | DEEPSEEK]"  # Prefix for logging
    ALIAS = "deepseek"  # Alias for filenames
    ALIAS_UPPER = "DEEPSEEK"  # Alias for logging

    def __init__(
        self,
        deepseek_model_name: str = "deepseek-r1:14b"
    ):
        """
        Initialize the Deepseek keyword extractor.
        """
        print(f"{self.STR_PREFIX} Initializing Deepseek keyword extractor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.deepseek_model_name = deepseek_model_name

        # Register the cleanup function to stop the model when the object is deleted
        atexit.register(self.stop_model)

        print("Done.")

    # Override
    def execute(self) -> List[str]:
        """
        Execute the LLM keyword extraction with DeepSeek.
        """
        print(f"{self.STR_PREFIX} Running LLM keyword extraction with DeepSeek...", flush=True)

        tags = self.execute_keyword_extraction()
        return tags
    
    def remove_thoughts(self, text: str) -> str:
        """
        Removes the <think></think> part of the response.
        """
        return text.split("</think>")[1]
    
    # Override
    def chat_llm(self, prompt: str) -> str:
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
    
    def stop_model(self):
        """
        Stop the DeepSeek model.
        """
        print(f"{self.STR_PREFIX} Stopping DeepSeek model...")
        subprocess.run(["ollama", "stop", self.deepseek_model_name])
        print("Done.")
    

def main():
    """
    Main function to run the DeepSeek keyword extractor.
    """
    keyword_extractor = DeepseekKeywordExtractor()

    keyword_extractor.load_inputs()

    tags = keyword_extractor.execute()

    keyword_extractor.save_outputs(tags)


if __name__ == "__main__":
    main()