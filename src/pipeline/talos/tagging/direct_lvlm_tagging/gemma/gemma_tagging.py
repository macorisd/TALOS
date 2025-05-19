import torch
from typing import List
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from dotenv import load_dotenv

from pipeline.talos.tagging.direct_lvlm_tagging.base_direct_lvlm_tagging import BaseDirectLvlmTagger

class GemmaTagger(BaseDirectLvlmTagger):
    """
    [Tagging -> Direct Tagging with LVLM -> Gemma]

    Tagging stage implementation that leverages the Gemma 3 model.
    """

    STR_PREFIX = "\n[TAGGING | GEMMA]"  # Prefix for logging
    ALIAS = "gemma"  # Alias for filenames

    def __init__(
        self,
        gemma_model_name: str = "google/gemma-3-27b-it",
        temperature: float = 0.5
    ):
        """
        Initialize the Gemma tagger.
        """
        print(f"{self.STR_PREFIX} Initializing Gemma tagger...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Variables
        self.temperature = temperature

        # Login to Hugging Face Hub
        load_dotenv()
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

        # Load model and processor
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_model_name, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(gemma_model_name)

        print("Done.")

    # Override from ITaggingStrategy -> BaseTagger -> BaseDirectLvlmTagger
    def execute(self) -> List[str]:
        """
        Execute the Gemma Tagging.
        """
        print(f"{self.STR_PREFIX} Running Tagging with Gemma...", flush=True)

        tags = self.execute_direct_lvlm_tagging()

        # Clear the GPU cache
        torch.cuda.empty_cache()

        return tags

    # Override from BaseDirectLvlmTagger
    def chat_lvlm(self) -> str:
        """
        Prompt the Gemma model with the input image and text prompt.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.input_image},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature
            )
            output_ids = generation[0][input_len:]

        output_text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text

def main():
    """
    Main function to run the Tagging stage with Gemma.
    """
    tagger = GemmaTagger()

    input_image_name = "avocado.jpeg"  # Replace with your actual image
    tagger.load_inputs(input_image_name)
    tags = tagger.execute()

    tagger.save_outputs(tags)

if __name__ == "__main__":
    main()
