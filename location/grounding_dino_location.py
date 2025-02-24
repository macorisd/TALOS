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

    def __init__(
            self,
            grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base",
            input_image_name: str = "dogs.jpg",            
    ):
        """
        TODO
        """
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
            raise FileNotFoundError(f"The image {input_image_name} was not found at {input_image_path}.")
        
        # Input keywords directory path
        self.input_keywords_dir = os.path.join(
            self.script_dir, 
            "..",                
            "tagging",
            "lvlm_llm_tagging",
            "llm_keywording",
            "output_keywords"
        )

        # Load keywords from the most recent .txt file in input_keywords_dir
        self.input_keywords = self.read_keywords_from_file()

        # Output location directory path
        self.output_location_dir = os.path.join(
            self.script_dir, 
            "output_location"
        )

        # Create the output location directory
        os.makedirs(self.output_location_dir, exist_ok=True)

    def read_keywords_from_file(self) -> str:
        """
        Reads the keywords from the most recent .txt file in the keywords directory.
        """
        # Gather all .txt files in input_keywords_dir
        txt_files = [
            os.path.join(self.input_keywords_dir, f)
            for f in os.listdir(self.input_keywords_dir)
            if f.endswith(".txt")
        ]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.input_keywords_dir}")

        # Select and read the most recently modified .txt file
        latest_txt_path = max(txt_files, key=os.path.getmtime)
        
        with open(latest_txt_path, "r", encoding="utf-8") as f:
            keywords_content = f.read()
        
        return keywords_content
    
    def json_to_gd_prompt(self, keywords: str) -> str:
        """
        Converts the keywords JSON text to a Grounding Dino prompt.
        """
        # Remove special characters from the keywords
        keywords = re.sub(r"[^a-zA-Z0-9\s]", "", keywords)

        # Split the keywords into a list
        keywords_list = keywords.split()

        # Build the Grounding Dino prompt
        prompt = ". ".join(keywords_list) + "."

        return prompt
    
    def locate_objects(self) -> dict:
        """
        TODO
        """
        # Convert the keywords JSON text to a Grounding Dino prompt
        text = self.json_to_gd_prompt(self.input_keywords)

        # Process and predict
        inputs = self.processor(images=self.input_image, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.input_image.size[::-1]]
        )[0]

        # Save the results to a text file 
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"location_gdino_{timestamp}.txt"
        output_file = os.path.join(self.output_location_dir, output_filename)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(results))
            print(f"[LOCATION | GDINO] Text results saved to: {output_file}")

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
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"location_gdino_{timestamp}.jpg"
        output_file = os.path.join(self.output_location_dir, output_filename)

        image.save(output_file)
        print(f"[LOCATION | GDINO] Result image saved to: {output_file}")

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