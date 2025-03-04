import time
from utils.print_utils import print_green, print_purple

from tagging.ram_plus_tagging.ram_plus_tagging import RamPlusTagger
from tagging.lvlm_llm_tagging.lvlm_description.llava_description import LlavaDescriptor
from tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek_keyword_extraction import DeepseekKeywordExtractor
from location.grounding_dino_location import GroundingDinoLocator
from segmentation.sam2_segmentation import Sam2Segmenter

RAM_PLUS = "[PIPELINE | TAGGING | RAM++]"
LVLM_LLM = "[PIPELINE | TAGGING | DESCRIPTION & KEYWORD EXTRACTION]"
LLAVA = "[PIPELINE | TAGGING | DESCRIPTION | LLAVA]"
DEEPSEEK = "[PIPELINE | TAGGING | KEYWORD EXTRACTION | DEEPSEEK]"
GROUNDING_DINO = "[PIPELINE | LOCATION | GDINO]"
SAM2 = "[PIPELINE | SEGMENTATION | SAM2]"

# TAGGING -------------------------------------------------------------------------------------

def tagging(input_image_name: str, tagging_method: str, tagging_submethods: tuple[str, str]) -> dict:

    # RAM++

    def tagging_ram_plus() -> dict:
        tagger = RamPlusTagger()
        tagger.load_image(input_image_name=input_image_name)
        tags_json = tagger.run()
        return tags_json

    # LLVM-LLM | LLAVA

    def description_llava() -> str:
        descriptor = LlavaDescriptor()
        descriptor.load_image_path(input_image_name=input_image_name)
        description_str = descriptor.run()
        return description_str

    # LLVM-LLM | DEEPSEEK

    def keyword_extraction_deepseek(pipeline_description: str) -> dict:
        extractor = DeepseekKeywordExtractor()
        extractor.load_description(pipeline_description=pipeline_description)
        tags_json = extractor.run()
        return tags_json

    # TAGGING

    if tagging_method == RAM_PLUS:
        print_green(f"{RAM_PLUS}\n")
        tags_json = tagging_ram_plus()
    elif tagging_method ==  LVLM_LLM:
        print_green(f"{LVLM_LLM}\n")
        if tagging_submethods[0] == LLAVA:
            print_green(f"{LLAVA}")
            description_str = description_llava()
        if tagging_submethods[1] == DEEPSEEK:
            print_green(f"{DEEPSEEK}")
            tags_json = keyword_extraction_deepseek(pipeline_description=description_str)

    return tags_json

# LOCATION -------------------------------------------------------------------------------------

def location(input_image_name: str, input_tags: dict, location_method: str) -> dict:

    # GROUNDING DINO

    def location_grounding_dino():
        locator = GroundingDinoLocator()
        locator.load_image(input_image_name=input_image_name)
        locator.load_tags(pipeline_tags=input_tags)
        location_output = locator.run()
        locator.draw_bounding_boxes(location_output)

    # LOCATION

    if location_method == GROUNDING_DINO:
        print_green(f"\n{GROUNDING_DINO}")
        location_grounding_dino()

# SEGMENTATION -------------------------------------------------------------------------------------

def segmentation(input_image_name: str, input_bbox_location: dict, segmentation_method: str):

    # SAM2

    def segmentation_sam2():
        segmenter = Sam2Segmenter()
        segmenter.load_image(input_image_name=input_image_name)
        segmenter.load_bbox_location(pipeline_bbox_location=input_bbox_location)
        segmenter.run()

    # SEGMENTATION

    if segmentation_method == SAM2:
        print_green(f"\n{SAM2}")
        segmentation_sam2()

# PIPELINE -------------------------------------------------------------------------------------

def pipeline(input_image_name: str, tagging_method: str, tagging_submethods: tuple[str, str], location_method: str, segmentation_method: str):
    start_time = time.time()

    print_purple(f"\n[PIPELINE] Starting pipeline execution...\n")

    tagging_output = tagging(
                        input_image_name=input_image_name, 
                        tagging_method=tagging_method, 
                        tagging_submethods=tagging_submethods
                    )
    
    location_output = location(
                        input_image_name=input_image_name, 
                        input_tags=tagging_output, 
                        location_method=location_method
                    )
    
    segmentation(
        input_image_name=input_image_name,
        input_bbox_location=location_output,
        segmentation_method=segmentation_method
    )

    end_time = time.time()
    print_purple(f"\n[PIPELINE] Pipeline execution completed in {end_time - start_time} seconds.\n")

def main():
    input_image_name = "4757.jpg"
    tagging_method = LVLM_LLM
    tagging_submethods = (LLAVA, DEEPSEEK)
    location_method = GROUNDING_DINO
    segmentation_method = SAM2

    pipeline(
        input_image_name=input_image_name, 
        tagging_method=tagging_method, 
        tagging_submethods=tagging_submethods, 
        location_method=location_method,
        segmentation_method=segmentation_method
    )

if __name__ == "__main__":
    main()