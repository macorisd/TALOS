import os
import torch
from PIL import Image
from typing import List

import warnings
warnings.filterwarnings("ignore", module="timm.models")

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

from pipeline.talos.tagging.base_tagging import BaseTagger
from pipeline.config.paths import INPUT_IMAGES_DIR

class RamPlusTagger(BaseTagger):
    """
    [Tagging -> Direct Tagging -> RAM++]
    
    Tagging stage implementation (Direct Tagging) that leverages the Recognize Anything Plus (RAM++) model.
    """
    STR_PREFIX = "\n[TAGGING | RAM++]" # Prefix for logging
    ALIAS = "ram_plus" # Alias for filenames

    def __init__(
        self,
        ram_plus_model_name: str = "ram_plus_swin_large_14m.pth",
        image_size: int = 384
    ):
        """
        Initialize the RAM++ tagger.
        """
        print(f"{self.STR_PREFIX} Initializing RAM++ tagger...", end=" ", flush=True)

        # Initialize base class
        super().__init__()

        # Load the model
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        ram_plus_model_path = os.path.join(self.script_dir, 'models', ram_plus_model_name)        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.transform = get_transform(image_size=image_size)
        self.model = ram_plus(pretrained=ram_plus_model_path, image_size=image_size, vit='swin_l').eval().to(self.device)

        print("Done.")
    
    # Override from ITaggingStrategy -> BaseTagger
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image for Direct Tagging with the RAM++ model.
        """
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name}...", end=" ", flush=True)

        # Input image path
        image_path = os.path.join(INPUT_IMAGES_DIR, input_image_name)

        # Load and transform input image
        if os.path.isfile(image_path):
            self.input_image = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image '{input_image_name}' was not found at {image_path}.")
        
        print("Done.")

    def __ram_tags_to_list(self, tags: str) -> List[str]:
        """
        Convert RAM++ tags to a list.
        """
        # Split the tags by the pipe character and strip any extra whitespace
        tags = tags.split('|')
        tags = [tag.strip() for tag in tags]
        
        return tags

    # Override from ITaggingStrategy -> BaseTagger
    def execute(self) -> List[str]:
        """
        Execute the RAM++ Tagging.
        """
        print(f"{self.STR_PREFIX} Running Direct Tagging with RAM++...", flush=True)
        tags = None

        with torch.no_grad():
            # Generate tags
            tags = inference(self.input_image, self.model)[0]
        
        print(f"{self.STR_PREFIX} Image tags: {tags}")

        # Convert RAM++ tags to str list
        tags_list = self.__ram_tags_to_list(tags)

        return tags_list
    

def main():
    """
    Main function to run the Direct Tagging stage with RAM++.
    """
    tagger = RamPlusTagger()
    
    input_image_name = "avocado.jpeg"
    tagger.load_inputs(input_image_name)
    tags = tagger.execute()
    
    tagger.save_outputs(tags)


if __name__ == "__main__":
    main()