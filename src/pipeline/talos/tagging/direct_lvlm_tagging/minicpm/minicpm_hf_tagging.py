import torch
from typing import List
import numpy as np

from transformers import AutoModel, AutoTokenizer

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
        minicpm_model_name: str = "openbmb/MiniCPM-o-2_6",
        temperature: float = 0.5
    ):
        """
        Initialize the MiniCPM tagger.
        """
        print(f"{self.STR_PREFIX} Initializing MiniCPM tagger...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Variables
        self.temperature = temperature

        self.model = AutoModel.from_pretrained(
            minicpm_model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False
        )

        # Establecer el modelo en modo de evaluación y moverlo a la GPU
        self.model = self.model.eval().cuda()

        # Cargar el tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(minicpm_model_name, trust_remote_code=True)

        print("Done.")
    
    # Override from ITaggingStrategy -> BaseTagger -> BaseDirectLvlmTagger
    def execute(self) -> List[str]:
        """
        Execute the MiniCPM Tagging.
        """
        print(f"{self.STR_PREFIX} Running Tagging with MiniCPM...", flush=True)

        tags = self.execute_direct_lvlm_tagging()

        # Clear the GPU cache
        torch.cuda.empty_cache()

        return tags

    def get_image_content(image, flatten=True):
        image_np = np.array(image)
        
        contents = []
        # Si `flatten` es True, agrega una lista plana
        if flatten:
            contents.extend(["<unit>", image_np])
        else:
            contents.append(["<unit>", image_np])
        
        return contents

    # Override from BaseDirectLvlmTagger
    def chat_lvlm(self) -> str:
        """
        Prompt the MiniCPM model with the input image and text prompt.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    self.input_image,
                    self.prompt
                ],
            }
        ]

        response = self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer
        )

        return response

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