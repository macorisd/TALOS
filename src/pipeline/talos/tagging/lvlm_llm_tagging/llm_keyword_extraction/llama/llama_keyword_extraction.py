from typing import List
import torch
import transformers

from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.base_keyword_extraction import BaseLlmKeywordExtractor

class LlamaKeywordExtractor(BaseLlmKeywordExtractor):
    """
    [Tagging -> LVLM + LLM -> LLM Keyword Extraction -> Llama]
    
    LLM keyword extraction implementation for the Tagging stage with the LVLM + LLM method that leverages the Llama model.
    """
    STR_PREFIX = "\n[TAGGING | LLM KEYWORD EXTRACTION | LLAMA]"  # Prefix for logging
    ALIAS = "llama"  # Alias for filenames
    ALIAS_UPPER = "LLAMA"  # Alias for logging

    def __init__(
        self,
        llama_model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        temperature: float = 0.6,
        top_p: float = 0.95
    ):
        """
        Initialize the Llama keyword extractor.
        """
        print(f"{self.STR_PREFIX} Initializing Llama keyword extractor...", end=" ")

        # Initialize base class
        super().__init__()

        # Variables
        self.llama_model_name = llama_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model configuration
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        
        # Load tokenizer and set pad token
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.llama_model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Create pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.llama_model_name,
            tokenizer=self.tokenizer,
            max_new_tokens=32768,
            temperature=self.temperature,
            top_p=self.top_p,
            **model_kwargs
        )

        print("Done.")

    # Override from ITaggingLlmStrategy -> BaseLlmKeywordExtractor
    def execute(self) -> List[str]:
        """
        Execute the LLM keyword extraction with Llama.
        """
        print(f"{self.STR_PREFIX} Running LLM keyword extraction with Llama...", flush=True)

        tags = self.execute_keyword_extraction()
        return tags

    def __parse_response(self, response_text: str) -> str:
        """
        Parse the response to extract the main content after thinking.
        For Llama Nemotron, the thinking content is typically enclosed in special tokens.
        """
        # The model may include thinking patterns, we need to extract the final answer
        # Look for patterns that indicate the end of thinking and start of answer
        lines = response_text.split('\n')
        
        # Find where the actual response starts (after thinking)
        start_index = 0
        for i, line in enumerate(lines):
            # Look for indicators that thinking has ended
            if any(indicator in line.lower() for indicator in ['answer:', 'solution:', 'result:', 'keywords:', 'tags:']):
                start_index = i
                break
            # Also check if line starts with actual content (not thinking markers)
            elif line.strip() and not line.startswith('thinking') and not line.startswith('<') and not line.startswith('['):
                # If we find structured content that looks like our expected output
                if '[' in line or '{' in line or line.strip().startswith('"'):
                    start_index = i
                    break
        
        # Join the relevant lines
        content = '\n'.join(lines[start_index:]).strip()
        return content

    # Override from BaseLlmKeywordExtractor
    def chat_llm(self, prompt: str) -> str:
        """
        Prompt the Llama model with the input text prompt.

        This method will be called by the superclass.
        """
        # Enable thinking mode for better reasoning
        thinking = "on"
        
        messages = [
            {"role": "system", "content": f"detailed thinking {thinking}"},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response using the pipeline
        response = self.pipeline(messages)
        
        # Extract the generated text
        if isinstance(response, list) and len(response) > 0:
            response_text = response[0].get('generated_text', '')
            
            # The response contains the full conversation, we need to extract only the assistant's response
            # Find the last assistant message
            if isinstance(response_text, list):
                # If response_text is a list of messages, get the last one
                assistant_response = response_text[-1].get('content', '') if response_text else ''
            else:
                # If it's a string, we need to parse it to find the assistant's response
                # This is a simplified approach - in practice, the format may vary
                assistant_response = response_text
        else:
            assistant_response = ""

        # Parse the response to extract meaningful content
        parsed_response = self.__parse_response(assistant_response)
        return parsed_response.strip()


def main():
    """
    Main function to run the Llama keyword extractor.
    """
    keyword_extractor = LlamaKeywordExtractor()

    keyword_extractor.load_inputs()

    tags = keyword_extractor.execute()

    keyword_extractor.save_outputs(tags)


if __name__ == "__main__":
    main()
