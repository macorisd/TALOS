import json
import os
from typing import Any

from pipeline.config.paths import CONFIG_DIR

# Available models for the TALOS pipeline

# Tagging models
RAM_PLUS = "ram_plus"
QWEN = "qwen"
LLAVA = "llava"
DEEPSEEK = "deepseek"

# Location models
GROUNDING_DINO = "grounding_dino"

# Segmentation models
SAM2 = "sam2"


# Available configuration parameters for the TALOS pipeline

# Pipeline models
PIPELINE_TAGGING = "PIPELINE_TAGGING"
PIPELINE_LOCATION = "PIPELINE_LOCATION"
PIPELINE_SEGMENTATION = "PIPELINE_SEGMENTATION"

# General configuration keys
SAVE_FILES = "SAVE_FILES"

# Tagging configuration keys
TAGGING_DIRECT_LVLM_TIMEOUT = "TAGGING_DIRECT_LVLM_TIMEOUT"
TAGGING_DIRECT_LVLM_ITERS = "TAGGING_DIRECT_LVLM_ITERS"
TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS = "TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS"
TAGGING_DIRECT_LVLM_BANNED_WORDS = "TAGGING_DIRECT_LVLM_BANNED_WORDS"

TAGGING_LVLM_TIMEOUT = "TAGGING_LVLM_TIMEOUT"
TAGGING_LVLM_ITERS = "TAGGING_LVLM_ITERS"

TAGGING_LLM_TIMEOUT = "TAGGING_LLM_TIMEOUT"
TAGGING_LLM_EXCLUDE_BANNED_WORDS = "TAGGING_LLM_EXCLUDE_BANNED_WORDS"
TAGGING_LLM_BANNED_WORDS = "TAGGING_LLM_BANNED_WORDS"
TAGGING_LLM_ENHANCE_OUTPUT = "TAGGING_LLM_ENHANCE_OUTPUT"

# Location configuration keys
LOCATION_SCORE_THRESHOLD = "LOCATION_SCORE_THRESHOLD"
LOCATION_PADDING_RATIO = "LOCATION_PADDING_RATIO"
LOCATION_LARGE_BBOX_RATIO = "LOCATION_LARGE_BBOX_RATIO"

# Segmentation configuration keys
# None for now


# Configuration paths in the config.json file

CONFIG_PATHS = {
    # Pipeline models
    PIPELINE_TAGGING: ["pipeline", "tagging"],
    PIPELINE_LOCATION: ["pipeline", "location"],
    PIPELINE_SEGMENTATION: ["pipeline", "segmentation"],
    # General configuration
    SAVE_FILES: ["general_config", "save_files"],
    # Tagging configuration
    TAGGING_DIRECT_LVLM_TIMEOUT: ["tagging", "direct_lvlm_tagging", "timeout"],
    TAGGING_DIRECT_LVLM_ITERS: ["tagging", "direct_lvlm_tagging", "iters"],
    TAGGING_DIRECT_LVLM_EXCLUDE_BANNED_WORDS: ["tagging", "direct_lvlm_tagging", "exclude_banned_words"],
    TAGGING_DIRECT_LVLM_BANNED_WORDS: ["tagging", "direct_lvlm_tagging", "banned_words"],
    TAGGING_LVLM_TIMEOUT: ["tagging", "lvlm_llm_tagging", "lvlm", "timeout"],
    TAGGING_LVLM_ITERS: ["tagging", "lvlm_llm_tagging", "lvlm", "iters"],
    TAGGING_LLM_TIMEOUT: ["tagging", "lvlm_llm_tagging", "llm", "timeout"],
    TAGGING_LLM_EXCLUDE_BANNED_WORDS: ["tagging", "lvlm_llm_tagging", "llm", "exclude_banned_words"],
    TAGGING_LLM_BANNED_WORDS: ["tagging", "lvlm_llm_tagging", "llm", "banned_words"],
    TAGGING_LLM_ENHANCE_OUTPUT: ["tagging", "lvlm_llm_tagging", "llm", "enhance_output"],
    # Location configuration
    LOCATION_SCORE_THRESHOLD: ["location", "score_threshold"],
    LOCATION_PADDING_RATIO: ["location", "padding_ratio"],
    LOCATION_LARGE_BBOX_RATIO: ["location", "large_bbox_ratio"],
    # Segmentation configuration
    # None for now
}


class ConfigSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file='config.json'):
        if not hasattr(self, 'initialized'):
            self.config = self.load_config(config_file)
            self.initialized = True
    
    def load_config(self, config_file):
        try:
            config_file = os.path.join(CONFIG_DIR, config_file)
            with open(config_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    def get(self, key: str) -> Any:
        if key not in CONFIG_PATHS:
            raise KeyError(f"Key '{key}' not found in configuration.")
        path = CONFIG_PATHS[key]
        ref = self.config
        for p in path:
            ref = ref[p]
        if ref is None:
            raise ValueError(f"Value for key '{key}' is None.")
        return ref

    def set(self, key: str, value: Any) -> None:
        if key not in CONFIG_PATHS:
            raise KeyError(f"Key '{key}' not found in configuration.")
        path = CONFIG_PATHS[key]
        ref = self.config
        for p in path[:-1]:
            ref = ref[p]
        ref[path[-1]] = value


config = ConfigSingleton()
