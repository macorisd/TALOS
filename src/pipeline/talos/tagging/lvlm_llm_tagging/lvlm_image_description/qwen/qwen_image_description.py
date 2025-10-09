import torch
from typing import List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.base_image_description import BaseLvlmImageDescriptor

class QwenImageDescriptor(BaseLvlmImageDescriptor):
    """
    [Tagging -> LVLM + LLM -> LVLM Image Description -> Qwen]
    
    LVLM image description implementation for the Tagging stage with the LVLM + LLM method that leverages the Qwen model.
    """

    STR_PREFIX = "\n[TAGGING | LVLM IMAGE DESCRIPTION | QWEN]" # Prefix for logging
    ALIAS = "qwen"  # Alias for filenames
    ALIAS_UPPER = "QWEN"  # Alias for logging

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        temperature: float = 0.5
    ):
        """
        Initialize the Qwen image descriptor.
        """
        print(f"{self.STR_PREFIX} Initializing Qwen image descriptor...", end=" ")

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
    
    # Override from ITaggingLvlmStrategy -> BaseLvlmImageDescriptor
    def execute(self) -> List[str]:
        """
        Execute the LVLM image description with Qwen.
        """
        print(f"{self.STR_PREFIX} Running LVLM image description with Qwen...", flush=True)

        descriptions = self.execute_image_description()
        return descriptions

    # Override from BaseLvlmImageDescriptor
    def chat_lvlm(self) -> str:
        """
        Prompt the Qwen model with the input image and text prompt.
        """
        input_image = self.input_image if hasattr(self, "input_image") else self.input_image_path

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_image
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
    Main function to run the Qwen image descriptor.
    """
    image_descriptor = QwenImageDescriptor()

    input_image_name = "avocado.jpeg"
    image_descriptor.load_inputs(input_image_name)

    descriptions = image_descriptor.execute()

    image_descriptor.save_outputs(descriptions)

if __name__ == "__main__":
    main()