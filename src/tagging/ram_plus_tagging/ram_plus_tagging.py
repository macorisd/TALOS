import warnings
warnings.filterwarnings("ignore", module="timm.models")

import os
import torch
import time
from PIL import Image
import json
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

class RamPlusTagger:
    """
    A class to extract tags from an image using the Recognize Anything Plus (RAM++) model.
    """

    STR_PREFIX = "\n[TAGGING | RAM++]"

    def __init__(
        self,        
        ram_plus_model_name: str = "ram_plus_swin_large_14m.pth",        
        image_size: int = 384,
        save_file: bool = True,
        timeout: int = 120
    ):
        """
        Initialize RAM++ tagger.
        """
        print(f"{self.STR_PREFIX} Initializing RAM++ tagger...", end=" ", flush=True)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))         
        self.save_file = save_file
        self.timeout = timeout if timeout > 0 else 120

        # Load the model
        ram_plus_model_path = os.path.join(self.script_dir, 'models', ram_plus_model_name)        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.transform = get_transform(image_size=image_size)
        self.model = ram_plus(pretrained=ram_plus_model_path, image_size=image_size, vit='swin_l').eval().to(self.device)

        if save_file:
            # Output directory for tags
            self.output_tags_dir = os.path.join(
                self.script_dir, 
                '..', 
                'output_tags'
            )

            # Create the output directory if it does not exist
            os.makedirs(self.output_tags_dir, exist_ok=True)   

        print("Done.")         

    def load_image(self, input_image_name: str) -> None:
        print(f"{self.STR_PREFIX} Loading input image: {input_image_name} ...", end=" ", flush=True)

        # Input image path
        image_path = os.path.join(
            self.script_dir, 
            "..", 
            "..",
            "input_images",
            input_image_name
        )

        # Load and transform input image
        if os.path.isfile(image_path):
            self.image = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image '{input_image_name}' was not found at {image_path}.")
        
        print("Done.")

    def ram_tags_to_list(self, tags: str) -> list:
        """
        Convert RAM++ tags to a list.
        """
        # Split the tags by the pipe character and strip any extra whitespace
        tags = tags.split('|')
        tags = [tag.strip() for tag in tags]
        
        return tags

    def run(self) -> list:
        """
        Generate tags from the input image.
        """
        start_time = time.time()
        tags = None

        while time.time() - start_time < self.timeout:
            with torch.no_grad():
                # Generate tags
                res = inference(self.image, self.model)
                if res and res[0].strip():
                    tags = res[0]
                    break
                print(f"{self.STR_PREFIX} No valid tags generated. Retrying...")
        else:
            raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving valid tags.")

        tags_list = self.ram_tags_to_list(tags)
        print(f"{self.STR_PREFIX} Image tags: {tags}")

        # Save tags to file
        if self.save_file:
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"tags_ram_plus_{timestamp}.json"
            output_file = os.path.join(self.output_tags_dir, output_filename)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tags_list, f, ensure_ascii=False, indent=4)
            print(f"{self.STR_PREFIX} Tags saved to {output_file}")
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Tags were not saved.")

        return tags_list


def main():
    tagger = RamPlusTagger()
    tagger.load_image(input_image_name="desk.jpg")
    tagger.run()


if __name__ == '__main__':
    main()