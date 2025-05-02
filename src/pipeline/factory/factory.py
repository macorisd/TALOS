from typing import Union, List
from strategy.strategy import *
from config.config import (
    RAM_PLUS,
    LLAVA,
    DEEPSEEK,
    GROUNDING_DINO,
    SAM2
)
from talos.tagging.direct_tagging.ram_plus.ram_plus_tagging import RamPlusTagger
from talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek.deepseek_keyword_extraction import DeepseekKeywordExtractor
from talos.tagging.lvlm_llm_tagging.lvlm_image_description.llava.llava_image_description import LlavaImageDescriptor
from talos.tagging.lvlm_llm_tagging.lvlm_llm_tagging import LvlmLlmTagger
from talos.location.grounding_dino.grounding_dino_location import GroundingDinoLocator
from talos.segmentation.sam2.sam2_segmentation import Sam2Segmenter


class StrategyFactory:
    @staticmethod
    def create_tagging_strategy(method: Union[str, List[str]]) -> ITaggingStrategy:
        if isinstance(method, str):
            if method == RAM_PLUS:
                return RamPlusTagger()
            else:
                raise ValueError(f"Unknown direct Tagging method: {method}")
        elif isinstance(method, list) and len(method) == 2:
            return LvlmLlmTagger(
                StrategyFactory.create_tagging_lvlm_strategy(method[0]),
                StrategyFactory.create_tagging_llm_strategy(method[1])
            )
        else:
            raise ValueError(f"Unknown Tagging method: {method}")

    @staticmethod
    def create_tagging_lvlm_strategy(lvlm_descriptor: str) -> ITaggingLvlmStrategy:
        if lvlm_descriptor == LLAVA:
            return LlavaImageDescriptor()
        else:
            raise ValueError(f"Unknown LVLM descriptor for LVLM + LLM Tagging: {lvlm_descriptor}")

    @staticmethod
    def create_tagging_llm_strategy(llm_keyword_extractor: str) -> ITaggingLlmStrategy:
        if llm_keyword_extractor == DEEPSEEK:
            return DeepseekKeywordExtractor()
        else:
            raise ValueError(f"Unknown LLM keyword extractor for LVLM + LLM Tagging: {llm_keyword_extractor}")
        
    @staticmethod
    def create_location_strategy(method: str) -> ILocationStrategy:
        if method == GROUNDING_DINO:
            return GroundingDinoLocator()
        else:
            raise ValueError(f"Unknown Location method: {method}")
        
    @staticmethod
    def create_segmentation_strategy(method: str) -> ISegmentationStrategy:
        if method == SAM2:
            return Sam2Segmenter()
        else:
            raise ValueError(f"Unknown Segmentation method: {method}")