import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import re
import time

class GroundingDinoLocator:
    """
    A class to locate objects in an image using the Grounding Dino model.
    """

    STR_PREFIX = "[LOCATION | GDINO]"

    def __init__(
            self,
            grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base",
            input_image_name: str = "input_image.jpg",            
    ):
        """
        TODO
        """

        print(f"\n{self.STR_PREFIX} Initializing Grounding DINO object locator...\n")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))        

        # Load the processor and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)

        # Input image path
        input_image_path = os.path.join(
            self.script_dir, 
            "..",                
            "input_images",
            input_image_name
        )

        # Load input image
        if os.path.isfile(input_image_path):
            self.input_image = Image.open(input_image_path)
        else:
            raise FileNotFoundError(f"{self.STR_PREFIX} The image {input_image_name} was not found at {input_image_path}.\n")
        
        # Input tags directory path
        input_tags_dir = os.path.join(
            self.script_dir, 
            "..",
            "tagging",
            "output_tags"
        )

        # Load tags from the most recent .txt file in input_tags_dir
        self.input_tags, tags_filename = self.read_input_tags(input_tags_dir=input_tags_dir)

        # Print input information
        print(f"{self.STR_PREFIX} Input image filename: {input_image_path}\n")
        print(f"{self.STR_PREFIX} Input tags filename: {input_tags_dir}/{tags_filename}\n")

        # Output location directory path
        output_location_dir = os.path.join(
            self.script_dir, 
            "output_location"
        )

        # Create the output directory if it does not exist
        os.makedirs(output_location_dir, exist_ok=True)

        # Prepare timestamped output file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename_txt = f"location_gdino_{timestamp}.txt"
        output_filename_jpg = f"location_gdino_{timestamp}.jpg"

        self.output_file_txt = os.path.join(output_location_dir, output_filename_txt)
        self.output_file_jpg = os.path.join(output_location_dir, output_filename_jpg)    

    def read_input_tags(self, input_tags_dir: str) -> tuple[str, str]:
        """
        Reads the tags from the most recent .txt file in the tags directory.

        Returns:
            tuple[str, str]: A tuple containing the content of the file and the filename.
        """
        # Gather all .txt files in input_tags_dir
        txt_files = [
            os.path.join(input_tags_dir, f)
            for f in os.listdir(input_tags_dir)
            if f.endswith(".txt")
        ]
        if not txt_files:
            raise FileNotFoundError(f"{self.STR_PREFIX} No .txt files found in {input_tags_dir}\n")

        # Select the most recently modified .txt file
        latest_txt_path = max(txt_files, key=os.path.getmtime)
        
        # Extract the filename (without the full path)
        filename = os.path.basename(latest_txt_path)
        
        # Read the content of the file
        with open(latest_txt_path, "r", encoding="utf-8") as f:
            tags_content = f.read()
        
        # Return a tuple of (content, filename)
        return tags_content, filename
    
    def json_to_gdino_prompt(self, tags: str) -> str:
        """
        Converts the tags JSON text to a Grounding DINO prompt.
        """
        # Remove special characters from the tags
        tags = re.sub(r"[^a-zA-Z0-9\s]", "", tags)

        # Split the tags into a list
        tags_list = tags.split()

        # Build the Grounding Dino prompt
        prompt = ". ".join(tags_list) + "."

        return prompt
    
    def locate_objects(self) -> dict:
        """
        TODO
        """
        # Convert the tags JSON text to a Grounding Dino prompt
        text = self.json_to_gdino_prompt(self.input_tags)

        # Process and predict
        inputs = self.processor(images=self.input_image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.input_image.size[::-1]]
        )[0]

        print(f"{self.STR_PREFIX} Object detection results:\n{results}\n")

        # Save the results to a text file
        with open(self.output_file_txt, "w", encoding="utf-8") as f:
            f.write(str(results))
            print(f"{self.STR_PREFIX} Text results saved to: {self.output_file_txt}\n")

        return results
    
    def draw_bounding_boxes(self, results: dict) -> Image:
        """
        Draws bounding boxes around the detected objects in the image.
        """    
        image = self.input_image.copy()
        draw = ImageDraw.Draw(image)        
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # Draw bounding boxes
        for score, box, label in zip(results["scores"], results["boxes"], results["text_labels"]):
            if score > 0.1:
                box = [int(b) for b in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red", font=font)
        
        # Save the image with bounding boxes
        image.save(self.output_file_jpg)
        print(f"{self.STR_PREFIX} Bounding box location image saved to: {self.output_file_jpg}")

        return image

def main():
    """
    Main function for the Grounding Dino Locator.
    """
    locator = GroundingDinoLocator(input_image_name="desk.jpg")
    location_output = locator.locate_objects()
    locator.draw_bounding_boxes(location_output)


if __name__ == "__main__":
    main()