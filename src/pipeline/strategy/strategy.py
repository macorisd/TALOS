from abc import ABC, abstractmethod
from typing import List, Dict


class ITaggingStrategy(ABC):
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def execute(self, image_path: str) -> List[str]:
        pass
    @abstractmethod
    def save_tags(self, tags: List[str]) -> None:
        pass

class ITaggingLvlmStrategy(ABC):
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def execute(self) -> List[str]:
        pass
    @abstractmethod
    def save_descriptions(self, descriptions: List[str]) -> None:
        pass

class ITaggingLlmStrategy(ABC):
    @abstractmethod
    def load_descriptions(self, descriptions: List[str]) -> None:
        pass
    @abstractmethod
    def execute(self) -> List[str]:
        pass

class ILocationStrategy(ABC):
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
    def save_location_json(self, location: Dict) -> None:
        pass
    @abstractmethod
    def save_location_image(self, location: Dict) -> None:
        pass

class ISegmentationStrategy(ABC):
    @abstractmethod
    def load_image(self, input_image_name: str) -> None:
        pass
    @abstractmethod
    def load_location(self, location: List[Dict]) -> None:
        pass
    @abstractmethod
    def execute(self) -> Dict:
        pass
