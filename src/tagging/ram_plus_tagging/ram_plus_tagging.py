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

    STR_PREFIX = "[TAGGING | RAM++]"

    def __init__(
        self,        
        ram_plus_model_name: str = "ram_plus_swin_large_14m.pth",
        input_image_name: str = "input_image.jpg",
        image_size: int = 384,
        save_file: bool = True,
        timeout: int = 120
    ):
        """
        Initialize RAM++ tagger.
        """
        print(f"\n{self.STR_PREFIX} Initializing RAM++ tagger...\n")
        print(f"{self.STR_PREFIX} Input image name: {input_image_name}\n")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))         
        self.save_file = save_file
        self.timeout = timeout

        # Load the model
        ram_plus_model_path = os.path.join(self.script_dir, 'models', ram_plus_model_name)        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.transform = get_transform(image_size=image_size)
        self.model = ram_plus(pretrained=ram_plus_model_path, image_size=image_size, vit='swin_l').eval().to(device)        

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
            self.image = self.transform(Image.open(image_path)).unsqueeze(0).to(device)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image '{input_image_name}' was not found at {image_path}.\n")

        # Output directory for tags
        output_tags_dir = os.path.join(
            self.script_dir, 
            '..', 
            'output_tags'
        )

        # Create the output directory if it does not exist
        os.makedirs(output_tags_dir, exist_ok=True)

        if save_file:
            # Prepare timestamped output file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"tags_ram_plus_{timestamp}.txt"
            self.output_file = os.path.join(output_tags_dir, output_filename)

    def ram_tags_to_json(self, tags: str) -> str:
        """
        Convert RAM++ tags to JSON format.
        """
        # Split the tags by the pipe character and strip any extra whitespace
        tags = tags.split('|')
        tags = [tag.strip() for tag in tags]
        
        # Create a dictionary with numbered keys
        tags_dict = {str(i+1): tag for i, tag in enumerate(tags)}
        
        # Convert the dictionary to a JSON string with indentation
        tags_json = json.dumps(tags_dict, indent=4, ensure_ascii=False)
        
        return tags_json

    def generate_tags(self) -> str:
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
                print(f"{self.STR_PREFIX} No valid tags generated. Retrying...\n")
        else:
            raise TimeoutError(f"{self.STR_PREFIX} Timeout of {self.timeout} seconds reached without receiving valid tags.\n")

        tags = self.ram_tags_to_json(tags)
        print(f"\n{self.STR_PREFIX} Image tags: {tags}\n")

        # Save tags to file
        if self.save_file:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(tags)
            print(f"{self.STR_PREFIX} Tags saved to {self.output_file}\n")
        else:
            print(f"{self.STR_PREFIX} Saving file is disabled. Tags were not saved.\n")

        return tags


def main():
    tagger = RamPlusTagger(
        input_image_name="desk.jpg"
    )

    tagger.generate_tags()


if __name__ == '__main__':
    main()