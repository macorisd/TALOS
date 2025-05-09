from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class ITaggingStrategy(ABC):
    @abstractmethod
    def load_inputs(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def execute(self, image_path: str) -> List[str]:
        pass
    @abstractmethod
    def save_outputs(self, tags: List[str]) -> None:
        pass
    @abstractmethod
    def save_tags(self, tags: List[str]) -> None:
        pass


class ITaggingLvlmStrategy(ABC):
    @abstractmethod
    def load_inputs(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def execute(self) -> List[str]:
        pass
    @abstractmethod
    def save_outputs(self, descriptions: List[str]) -> None:
        pass
    @abstractmethod
    def save_descriptions(self, descriptions: List[str]) -> None:
        pass


class ITaggingLlmStrategy(ABC):
    @abstractmethod
    def load_inputs(self, descriptions: List[str] = None) -> None:
        pass
    @abstractmethod
    def load_descriptions(self, descriptions: List[str]) -> None:
        pass
    @abstractmethod
    def execute(self) -> List[str]:
        pass
    @abstractmethod
    def save_outputs(self, tags: List[str]) -> None:
        pass
    @abstractmethod
    def save_tags(self, tags: List[str]) -> None:
        pass


class ILocationStrategy(ABC):
    @abstractmethod
    def load_inputs(self, input_image_name: str, input_tags: List[str]) -> None:
        pass
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def load_tags(self, input_tags: List[str]) -> None:
        pass
    @abstractmethod
    def execute(self) -> List[Dict]:
        pass
    @abstractmethod
    def save_outputs(self, location: Dict) -> None:
        pass
    @abstractmethod
    def save_location_json(self, location: Dict) -> None:
        pass
    @abstractmethod
    def save_location_image(self, location: Dict) -> None:
        pass


class ISegmentationStrategy(ABC):
    @abstractmethod
    def load_inputs(self, input_image_name: str, input_location: List[Dict] = None) -> None:
        pass
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def load_location(self, input_location: List[Dict]) -> None:
        pass
    @abstractmethod
    def execute(self) -> Dict:
        pass
    @abstractmethod
    def save_outputs(self, segmentation_info: Dict, all_masks: List[np.ndarray]) -> None:
        pass
    @abstractmethod
    def save_detections_json(self, segmentation_info: Dict) -> None:
        pass
    @abstractmethod
    def save_segmentation_masks(self, masks: List[np.ndarray]) -> None:
        pass
    @abstractmethod
    def save_segmentation_images(self, masks: List[np.ndarray]) -> None:
        pass
    @abstractmethod
    def save_segmentation_highlighted_images(self, masks: List[np.ndarray]) -> None:
        pass
