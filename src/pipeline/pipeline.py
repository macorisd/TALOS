import time
from utils.print_utils import print_green, print_purple

from tagging.ram_plus_tagging.ram_plus_tagging import RamPlusTagger
from tagging.lvlm_llm_tagging.lvlm_description.llava_description import LlavaDescriptor
from tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek_keyword_extraction import DeepseekKeywordExtractor
from location.grounding_dino_location import GroundingDinoLocator
from segmentation.sam2_segmentation import Sam2Segmenter

RAM_PLUS = "\n[PIPELINE | TAGGING | RAM++]"
LVLM_LLM = "\n[PIPELINE | TAGGING | DESCRIPTION & KEYWORD EXTRACTION]"
LLAVA = "\n[PIPELINE | TAGGING | DESCRIPTION | LLAVA]"
DEEPSEEK = "\n[PIPELINE | TAGGING | KEYWORD EXTRACTION | DEEPSEEK]"
GROUNDING_DINO = "\n[PIPELINE | LOCATION | GDINO]"
SAM2 = "\n[PIPELINE | SEGMENTATION | SAM2]"

class PipelineTLS:
    def __init__(
            self,
            tagging_method: str,
            tagging_submethods: str,
            location_method: str,
            segmentation_method: str,
            save_files: bool = False
        ):
        print_purple("\n[PIPELINE] Loading models...")

        self.tagging_method = tagging_method
        self.tagging_submethods = tagging_submethods
        self.location_method = location_method
        self.segmentation_method = segmentation_method
        self.save_files = save_files

        # Initialize models
        if tagging_method == RAM_PLUS:
            self.tagger_ram_plus = RamPlusTagger(save_file=save_files)
        elif tagging_method == LVLM_LLM:
            if tagging_submethods[0] == LLAVA:
                self.descriptor_llava = LlavaDescriptor(save_file=save_files, iters=1)
            if tagging_submethods[1] == DEEPSEEK:
                self.extractor_deepseek = DeepseekKeywordExtractor(save_file=save_files)

        if location_method == GROUNDING_DINO:
            self.locator_gdino = GroundingDinoLocator(save_file_jpg=save_files, save_file_json=save_files)
        
        if segmentation_method == SAM2:
            self.segmenter_sam2 = Sam2Segmenter()

        print_purple("\n[PIPELINE] All models loaded successfully.")

    def tagging(self, input_image_name: str) -> list:
        if self.tagging_method == RAM_PLUS:
            print_green(f"{RAM_PLUS}")
            self.tagger_ram_plus.load_image(input_image_name)
            return self.tagger_ram_plus.run()
                
        elif self.tagging_method == LVLM_LLM:
            print_green(f"{LVLM_LLM}")

            if self.tagging_submethods[0] == LLAVA:
                print_green(f"{LLAVA}")
                self.descriptor_llava.load_image_path(input_image_name)
                description_list = self.descriptor_llava.run()

            if self.tagging_submethods[1] == DEEPSEEK:
                print_green(f"{DEEPSEEK}")
                self.extractor_deepseek.load_description(description_list)
                return self.extractor_deepseek.run()
        return []

    def location(self, input_image_name: str, input_tags: dict) -> dict:
        if self.location_method == GROUNDING_DINO:
            print_green(f"{GROUNDING_DINO}")
            self.locator_gdino.load_image(input_image_name)
            self.locator_gdino.load_tags(input_tags)
            return self.locator_gdino.run()
        return {}

    def segmentation(self, input_image_name: str, input_bbox_location: dict):
        if self.segmentation_method == SAM2:
            print_green(f"{SAM2}")
            self.segmenter_sam2.load_image(input_image_name)
            self.segmenter_sam2.load_bbox_location(input_bbox_location)
            self.segmenter_sam2.run()

    def run(self, input_image_name: str) -> float:
        start_time = time.time()
        print_purple("\n[PIPELINE] Starting pipeline execution...")

        tagging_output = self.tagging(input_image_name)
        location_output = self.location(input_image_name, tagging_output)
        self.segmentation(input_image_name, location_output)

        end_time = time.time()
        total_time = end_time - start_time

        print_purple(f"\n[PIPELINE] Pipeline execution completed in {total_time} seconds.")
        return total_time


def main(iters: int = 1):
    tagging_method = LVLM_LLM
    tagging_submethods = (LLAVA, DEEPSEEK)
    location_method = GROUNDING_DINO
    segmentation_method = SAM2

    pipeline = PipelineTLS(
        tagging_method=tagging_method,
        tagging_submethods=tagging_submethods,
        location_method=location_method,
        segmentation_method=segmentation_method,
        save_files=True
    )

    # input_image_name = "desk.jpg"
    input_image_name = ["desk.jpg", "279.jpg", "603.jpg", "963.jpg", "1108.jpg", "1281.jpg", "1514.jpg", "1729.jpg", "1871.jpg", "2421.jpg"]

    # One iteration
    if iters <= 1:
        pipeline.run(input_image_name=input_image_name)
    
    # Multiple iterations, to measure the average execution time
    else:
        total_time = 0

        for i in range(iters):
            print_purple(f"\n[PIPELINE] Execution {i+1}/{iters}...")
            # total_time += pipeline.run(input_image_name=input_image_name)
            total_time += pipeline.run(input_image_name=input_image_name[i])

        avg_time = total_time / iters
        print_purple(f"\n[PIPELINE] Average execution time over {iters} runs: {avg_time} seconds.")

if __name__ == "__main__":
    main(2)