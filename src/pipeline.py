import time
from utils.print_utils import print_green, print_purple

from tagging.ram_plus_tagging.ram_plus_tagging import RamPlusTagger
from tagging.lvlm_llm_tagging.lvlm_description.llava_description import LlavaDescriptor
from tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek_keyword_extraction import DeepseekKeywordExtractor
from location.grounding_dino_location import GroundingDinoLocator

RAM_PLUS = "[PIPELINE | TAGGING | RAM++]"
LVLM_LLM = "[PIPELINE | TAGGING | DESCRIPTION & KEYWORD EXTRACTION]"
LLAVA = "[PIPELINE | TAGGING | DESCRIPTION | LLAVA]"
DEEPSEEK = "[PIPELINE | TAGGING | KEYWORD EXTRACTION | DEEPSEEK]"
GROUNDING_DINO = "[PIPELINE | LOCATION | GDINO]"

# TAGGING -------------------------------------------------------------------------------------

def tagging(input_image_name: str, tagging_method: str, tagging_submethods: tuple[str, str]):

    # RAM++

    def tagging_ram_plus():
        tagger = RamPlusTagger(
            input_image_name=input_image_name
        )
        tagger.generate_tags()

    # LLVM-LLM | LLAVA

    def description_llava():
        descriptor = LlavaDescriptor(input_image_name=input_image_name)    
        descriptor.describe_image()

    # LLVM-LLM | DEEPSEEK

    def keyword_extraction_deepseek():
        extractor = DeepseekKeywordExtractor()
        extractor.extract_keywords()

    # TAGGING

    if tagging_method == RAM_PLUS:
        print_green(f"{RAM_PLUS}\n")
        tagging_ram_plus()
    elif tagging_method ==  LVLM_LLM:
        print_green(f"{LVLM_LLM}\n")
        if tagging_submethods[0] == LLAVA:
            print_green(f"{LLAVA}")
            description_llava()
        if tagging_submethods[1] == DEEPSEEK:
            print_green(f"{DEEPSEEK}")
            keyword_extraction_deepseek()

# LOCATION -------------------------------------------------------------------------------------

def location(input_image_name: str, location_method: str):

    # GROUNDING DINO

    def location_grounding_dino():
        locator = GroundingDinoLocator(input_image_name=input_image_name)
        location_output = locator.locate_objects()
        locator.draw_bounding_boxes(location_output)

    # LOCATION

    if location_method == GROUNDING_DINO:
        print_green(f"\n{GROUNDING_DINO}")
        location_grounding_dino()

# PIPELINE -------------------------------------------------------------------------------------

def pipeline(input_image_name: str, tagging_method: str, tagging_submethods: tuple[str, str], location_method: str):
    start_time = time.time()

    print_purple(f"\n[PIPELINE] Starting pipeline execution...\n")

    tagging(input_image_name=input_image_name, tagging_method=tagging_method, tagging_submethods=tagging_submethods)
    location(input_image_name=input_image_name, location_method=location_method)

    end_time = time.time()
    print_purple(f"\n[PIPELINE] Pipeline execution completed in {end_time - start_time} seconds.\n")

def main():
    input_image_name = "desk.jpg"
    tagging_method = LVLM_LLM
    tagging_submethods = (LLAVA, DEEPSEEK)
    location_method = GROUNDING_DINO

    pipeline(input_image_name=input_image_name, tagging_method=tagging_method, tagging_submethods=tagging_submethods, location_method=location_method)

if __name__ == "__main__":
    main()