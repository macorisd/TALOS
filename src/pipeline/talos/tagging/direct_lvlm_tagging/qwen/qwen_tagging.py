import torch
from typing import List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

from pipeline.talos.tagging.direct_lvlm_tagging.base_direct_lvlm_tagging import BaseDirectLvlmTagger

class QwenTagger(BaseDirectLvlmTagger):
    """
    [Tagging -> Direct Tagging with LVLM -> Qwen]
    
    Tagging stage implementation that leverages the Qwen model.
    """

    STR_PREFIX = "\n[TAGGING | QWEN]" # Prefix for logging
    ALIAS = "qwen" # Alias for filenames

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        temperature: float = 0.5
    ):
        """
        Initialize the Qwen tagger.
        """
        print(f"{self.STR_PREFIX} Initializing Qwen tagger...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Variables
        self.temperature = temperature

        # Load the model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)

        print("Done.")
    
    # Override
    def execute(self) -> List[str]:
        """
        Execute the Qwen Tagging.
        """
        print(f"{self.STR_PREFIX} Running Tagging with Qwen...", flush=True)

        tags = self.execute_direct_lvlm_tagging()

        # Clear the GPU cache
        torch.cuda.empty_cache()

        return tags

    # Override
    def chat_lvlm(self) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.input_image,
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

def main():
    """
    Main function to run the Qwen tagger.
    """
    tagger = QwenTagger()
    
    input_image_name = "desk.jpg"
    tagger.load_inputs(input_image_name)
    tags = tagger.execute()
    
    tagger.save_outputs(tags)

if __name__ == "__main__":
    main()