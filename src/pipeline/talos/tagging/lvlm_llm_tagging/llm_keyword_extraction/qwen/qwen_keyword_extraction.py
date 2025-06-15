from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.base_keyword_extraction import BaseLlmKeywordExtractor

class QwenKeywordExtractor(BaseLlmKeywordExtractor):
    """
    [Tagging -> LVLM + LLM -> LLM Keyword Extraction -> Qwen]
    
    LLM keyword extraction implementation for the Tagging stage with the LVLM + LLM method that leverages the Qwen model.
    """
    STR_PREFIX = "\n[TAGGING | LLM KEYWORD EXTRACTION | QWEN]"  # Prefix for logging
    ALIAS = "qwen"  # Alias for filenames
    ALIAS_UPPER = "QWEN"  # Alias for logging

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen3-8B"
    ):
        """
        Initialize the Qwen keyword extractor.
        """
        print(f"{self.STR_PREFIX} Initializing Qwen keyword extractor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.qwen_model_name = qwen_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        print("Done.")

    # Override from ITaggingLlmStrategy -> BaseLlmKeywordExtractor
    def execute(self) -> List[str]:
        """
        Execute the LLM keyword extraction with Qwen.
        """
        print(f"{self.STR_PREFIX} Running LLM keyword extraction with Qwen...", flush=True)

        tags = self.execute_keyword_extraction()
        return tags

    def __remove_thoughts(self, output_ids) -> str:
        """
        Parse the output to extract the content after </think>.
        """
        try:
            # 151668 is the token id for </think>
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content

    # Override from BaseLlmKeywordExtractor
    def chat_llm(self, prompt: str) -> str:
        """
        Prompt the Qwen model with the input text prompt.

        This method will be called by the superclass.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response_content = self.__remove_thoughts(output_ids)
        return response_content.strip()


def main():
    """
    Main function to run the Qwen keyword extractor.
    """
    keyword_extractor = QwenKeywordExtractor()

    keyword_extractor.load_inputs()

    tags = keyword_extractor.execute()

    keyword_extractor.save_outputs(tags)


if __name__ == "__main__":
    main()