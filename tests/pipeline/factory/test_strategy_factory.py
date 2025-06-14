import unittest
from pipeline.factory.factory import StrategyFactory
from pipeline.strategy.strategy import (
    ITaggingStrategy,
    ILocationStrategy,
    ISegmentationStrategy,
    ITaggingLvlmStrategy,
    ITaggingLlmStrategy
)
from pipeline.config.config import (
    RAM_PLUS,
    QWEN,
    GEMMA,
    MINICPM,
    LLAVA,
    DEEPSEEK,
    LLAMA,
    GROUNDING_DINO,
    SAM2
)

from pipeline.talos.tagging.direct_lvlm_tagging.qwen.qwen_tagging import QwenTagger
from pipeline.talos.tagging.direct_lvlm_tagging.gemma.gemma_tagging import GemmaTagger
from pipeline.talos.tagging.direct_tagging.ram_plus.ram_plus_tagging import RamPlusTagger
from pipeline.talos.tagging.direct_lvlm_tagging.minicpm.minicpm_tagging import MiniCpmTagger
from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_llm_tagging import LvlmLlmTagger
from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.llava.llava_image_description import LlavaImageDescriptor
from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.qwen.qwen_image_description import QwenImageDescriptor
from pipeline.talos.tagging.lvlm_llm_tagging.lvlm_image_description.minicpm.minicpm_image_description import MiniCpmImageDescriptor
from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.deepseek.deepseek_keyword_extraction import DeepseekKeywordExtractor
from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.minicpm.minicpm_keyword_extraction import MiniCpmKeywordExtractor
from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.qwen.qwen_keyword_extraction import QwenKeywordExtractor
from pipeline.talos.tagging.lvlm_llm_tagging.llm_keyword_extraction.llama.llama_keyword_extraction import LlamaKeywordExtractor
from pipeline.talos.location.grounding_dino.grounding_dino_location import GroundingDinoLocator
from pipeline.talos.segmentation.sam2.sam2_segmentation import Sam2Segmenter

class TestStrategyFactory(unittest.TestCase):

    def test_create_tagging_strategy_direct_qwen(self):
        # Arrange
        method = QWEN
        # Act
        strategy = StrategyFactory.create_tagging_strategy(method)
        # Assert
        self.assertIsInstance(strategy, QwenTagger)
        self.assertIsInstance(strategy, ITaggingStrategy)

    def test_create_tagging_strategy_direct_gemma(self):
        # Arrange
        method = GEMMA
        # Act
        strategy = StrategyFactory.create_tagging_strategy(method)
        # Assert
        self.assertIsInstance(strategy, GemmaTagger)
        self.assertIsInstance(strategy, ITaggingStrategy)

    def test_create_tagging_strategy_direct_minicpm(self):
        # Arrange
        method = MINICPM
        # Act
        strategy = StrategyFactory.create_tagging_strategy(method)
        # Assert
        self.assertIsInstance(strategy, MiniCpmTagger)
        self.assertIsInstance(strategy, ITaggingStrategy)

    def test_create_tagging_strategy_direct_ram_plus(self):
        # Arrange
        method = RAM_PLUS
        # Act
        strategy = StrategyFactory.create_tagging_strategy(method)
        # Assert
        self.assertIsInstance(strategy, RamPlusTagger)
        self.assertIsInstance(strategy, ITaggingStrategy)

    def test_create_tagging_strategy_lvlm_llm(self): 
        # Arrange
        method = [LLAVA, DEEPSEEK]
        # Act
        strategy = StrategyFactory.create_tagging_strategy(method)
        # Assert
        self.assertIsInstance(strategy, LvlmLlmTagger)
        self.assertIsInstance(strategy, ITaggingStrategy)
        self.assertIsInstance(strategy.lvlm_image_descriptor, LlavaImageDescriptor)
        self.assertIsInstance(strategy.llm_keyword_extractor, DeepseekKeywordExtractor)

    def test_create_tagging_strategy_invalid_string(self):
        # Arrange
        method = "INVALID_METHOD"
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_tagging_strategy(method)

    def test_create_tagging_strategy_invalid_list_length(self):
        # Arrange
        method = [LLAVA] # Not length 2
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_tagging_strategy(method)

    def test_create_tagging_strategy_invalid_list_type(self):
        # Arrange
        method = ["INVALID_LVLM", "INVALID_LLM"]
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_tagging_strategy(method)

    def test_create_tagging_lvlm_strategy_llava(self):
        # Arrange
        lvlm_descriptor = LLAVA
        # Act
        strategy = StrategyFactory.create_tagging_lvlm_strategy(lvlm_descriptor)
        # Assert
        self.assertIsInstance(strategy, LlavaImageDescriptor)
        self.assertIsInstance(strategy, ITaggingLvlmStrategy)

    def test_create_tagging_lvlm_strategy_qwen(self):
        # Arrange
        lvlm_descriptor = QWEN
        # Act
        strategy = StrategyFactory.create_tagging_lvlm_strategy(lvlm_descriptor)
        # Assert
        self.assertIsInstance(strategy, QwenImageDescriptor)
        self.assertIsInstance(strategy, ITaggingLvlmStrategy)
        
    def test_create_tagging_lvlm_strategy_minicpm(self):
        # Arrange
        lvlm_descriptor = MINICPM
        # Act
        strategy = StrategyFactory.create_tagging_lvlm_strategy(lvlm_descriptor)
        # Assert
        self.assertIsInstance(strategy, MiniCpmImageDescriptor)
        self.assertIsInstance(strategy, ITaggingLvlmStrategy)

    def test_create_tagging_lvlm_strategy_invalid(self):
        # Arrange
        lvlm_descriptor = "INVALID_LVLM"
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_tagging_lvlm_strategy(lvlm_descriptor)

    def test_create_tagging_llm_strategy_deepseek(self):
        # Arrange
        llm_keyword_extractor = DEEPSEEK
        # Act
        strategy = StrategyFactory.create_tagging_llm_strategy(llm_keyword_extractor)
        # Assert
        self.assertIsInstance(strategy, DeepseekKeywordExtractor)
        self.assertIsInstance(strategy, ITaggingLlmStrategy)

    def test_create_tagging_llm_strategy_minicpm(self):
        # Arrange
        llm_keyword_extractor = MINICPM
        # Act
        strategy = StrategyFactory.create_tagging_llm_strategy(llm_keyword_extractor)
        # Assert
        self.assertIsInstance(strategy, MiniCpmKeywordExtractor)
        self.assertIsInstance(strategy, ITaggingLlmStrategy)

    def test_create_tagging_llm_strategy_qwen(self):
        # Arrange
        llm_keyword_extractor = QWEN
        # Act
        strategy = StrategyFactory.create_tagging_llm_strategy(llm_keyword_extractor)
        # Assert
        self.assertIsInstance(strategy, QwenKeywordExtractor)
        self.assertIsInstance(strategy, ITaggingLlmStrategy)

    def test_create_tagging_llm_strategy_llama(self):
        # Arrange
        llm_keyword_extractor = LLAMA
        # Act
        strategy = StrategyFactory.create_tagging_llm_strategy(llm_keyword_extractor)
        # Assert
        self.assertIsInstance(strategy, LlamaKeywordExtractor)
        self.assertIsInstance(strategy, ITaggingLlmStrategy)

    def test_create_tagging_llm_strategy_invalid(self):
        # Arrange
        llm_keyword_extractor = "INVALID_LLM"
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_tagging_llm_strategy(llm_keyword_extractor)

    def test_create_location_strategy_grounding_dino(self):
        # Arrange
        method = GROUNDING_DINO
        # Act
        strategy = StrategyFactory.create_location_strategy(method)
        # Assert
        self.assertIsInstance(strategy, GroundingDinoLocator)
        self.assertIsInstance(strategy, ILocationStrategy)

    def test_create_location_strategy_invalid(self):
        # Arrange
        method = "INVALID_LOCATION_METHOD"
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_location_strategy(method)

    def test_create_segmentation_strategy_sam2(self):
        # Arrange
        method = SAM2
        # Act
        strategy = StrategyFactory.create_segmentation_strategy(method)
        # Assert
        self.assertIsInstance(strategy, Sam2Segmenter)
        self.assertIsInstance(strategy, ISegmentationStrategy)

    def test_create_segmentation_strategy_invalid(self):
        # Arrange
        method = "INVALID_SEGMENTATION_METHOD"
        # Act & Assert
        with self.assertRaises(ValueError):
            StrategyFactory.create_segmentation_strategy(method)

if __name__ == '__main__':
    unittest.main()
