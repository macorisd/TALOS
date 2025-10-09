from typing import List
import numpy as np

from pipeline.talos.tagging.base_tagging import BaseTagger
from pipeline.strategy.strategy import (
    ITaggingLvlmStrategy,
    ITaggingLlmStrategy
)
from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek.deepseek_keyword_extraction import DeepseekKeywordExtractor
from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.llava.llava_image_description import LlavaImageDescriptor

class LvlmLlmTagger(BaseTagger):
    """
    [Tagging -> LVLM + LLM]
    
    Tagging stage implementation that combines LVLM image description with LLM keyword extraction.
    """
    def __init__(
        self,
        lvlm_image_descriptor: ITaggingLvlmStrategy,
        llm_keyword_extractor: ITaggingLlmStrategy
    ):
        """
        Initialize the LVLM + LLM tagger.
        """
        # Initialize base class
        super().__init__()

        # LVLM and LLM strategies
        self.lvlm_image_descriptor = lvlm_image_descriptor
        self.llm_keyword_extractor = llm_keyword_extractor

        # Logging prefix and alias
        self.STR_PREFIX = f"\n[TAGGING | LVLM + LLM | {self.lvlm_image_descriptor.ALIAS_UPPER} + {self.llm_keyword_extractor.ALIAS_UPPER}]"
        self.ALIAS = f"lvlm_llm_{self.lvlm_image_descriptor.ALIAS}_{self.llm_keyword_extractor.ALIAS}"


    # Override from ITaggingStrategy -> BaseTagger
    def load_image(self, input_image_name: str) -> None:
        """
        Load the input image's path in the LVLM image descriptor.
        """
        self.lvlm_image_descriptor.load_image(input_image_name)
    
    # Override from BaseTagger
    def set_image(self, input_image: np.ndarray) -> None:
        """
        Set the input image in the LVLM image descriptor.
        """
        self.lvlm_image_descriptor.set_image(input_image)

    # Override from ITaggingStrategy -> BaseTagger
    def execute(self) -> List[str]:
        """
        Execute the LVLM + LLM Tagging: LVLM image description and LLM keyword extraction.
        """
        print(f"{self.STR_PREFIX} Running LVLM + LLM Tagging...", flush=True)

        descriptions = self.lvlm_image_descriptor.execute()
        self.lvlm_image_descriptor.save_outputs(descriptions)

        self.llm_keyword_extractor.load_descriptions(descriptions)
        tags = self.llm_keyword_extractor.execute()

        return tags

def main():
    """
    Main function to run the LVLM + LLM Tagging.
    """
    # Initialize LVLM image descriptor and LLM keyword extractor
    lvlm_image_descriptor = LlavaImageDescriptor()
    llm_keyword_extractor = DeepseekKeywordExtractor()

    # Create LVLM + LLM tagger
    tagger = LvlmLlmTagger(lvlm_image_descriptor, llm_keyword_extractor)

    # Load input image
    input_image_name = "avocado.jpeg"  # Replace with actual image name
    tagger.load_inputs(input_image_name)

    # Execute tagging
    tags = tagger.execute()
    
    # Save outputs
    tagger.save_outputs(tags)

if __name__ == "__main__":
    main()