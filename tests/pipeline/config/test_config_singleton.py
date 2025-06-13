import unittest
import os
import json
from unittest.mock import patch, mock_open

from pipeline.config.config import ConfigSingleton, CONFIG_PATHS, PIPELINE_TAGGING, SAVE_INTERMEDIATE_FILES
from pipeline.config.paths import CONFIG_DIR

class TestConfigSingleton(unittest.TestCase):

    def setUp(self):
        # Reset singleton instance for each test to ensure isolation
        ConfigSingleton._instance = None
        self.mock_config_data = {
            "pipeline": {
                "tagging": "qwen",
                "location": "grounding_dino",
                "segmentation": "sam2"
            },
            "general_config": {
                "save_intermediate_files": True,
                "save_segmentation_files": False
            },
            "tagging": {
                "direct_lvlm_tagging": {
                    "timeout": 60,
                    "iters": 1,
                    "exclude_banned_words": True,
                    "banned_words": ["word1", "word2"]
                },
                "lvlm_llm_tagging": {
                    "lvlm": {
                        "timeout": 120,
                        "iters": 1
                    },
                    "llm": {
                        "timeout": 30,
                        "exclude_banned_words": False,
                        "banned_words": [],
                        "enhance_output": True
                    }
                }
            },
            "location": {
                "score_threshold": 0.5,
                "padding_ratio": 0.1,
                "large_bbox_ratio": 0.8
            }
        }

    def tearDown(self):
        ConfigSingleton._instance = None

    def test_singleton_instance(self):
        # Arrange
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            # Act
            instance1 = ConfigSingleton(config_file='mock_config.json')
            instance2 = ConfigSingleton(config_file='mock_config.json')
            # Assert
            self.assertIs(instance1, instance2)
            self.assertTrue(instance1.initialized)
            self.assertTrue(instance2.initialized)

    def test_load_config_success(self):
        # Arrange
        mock_json_data = json.dumps(self.mock_config_data)
        with patch('builtins.open', mock_open(read_data=mock_json_data)) as mocked_file:
            # Act
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Assert
            mocked_file.assert_called_once_with(os.path.join(CONFIG_DIR, 'mock_config.json'), 'r')
            self.assertEqual(config_singleton.config, self.mock_config_data)
            self.assertTrue(config_singleton.initialized)

    def test_load_config_file_not_found(self):
        # Arrange
        with patch('builtins.open', side_effect=FileNotFoundError):
            # Act & Assert
            with self.assertRaises(FileNotFoundError):
                ConfigSingleton(config_file='non_existent_config.json')

    def test_get_existing_key(self):
        # Arrange
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act
            value = config_singleton.get(PIPELINE_TAGGING)
            # Assert
            self.assertEqual(value, "qwen")

    def test_get_key_with_none_value_in_config_paths_but_not_none_in_file(self):
        # Arrange
        current_mock_config = self.mock_config_data.copy()
        CONFIG_PATHS["NEW_NONE_KEY"] = ["general_config", "new_none_param"]
        current_mock_config["general_config"]["new_none_param"] = None

        with patch('builtins.open', mock_open(read_data=json.dumps(current_mock_config))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act & Assert
            with self.assertRaisesRegex(ValueError, "Value for key 'NEW_NONE_KEY' is None."):
                config_singleton.get("NEW_NONE_KEY")
        del CONFIG_PATHS["NEW_NONE_KEY"]

    def test_get_non_existent_key_in_config_paths(self):
        # Arrange
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act & Assert
            with self.assertRaisesRegex(KeyError, "Key 'NON_EXISTENT_KEY' not found in configuration."):
                config_singleton.get("NON_EXISTENT_KEY")

    def test_get_key_path_not_in_file(self):
        # Arrange
        # Simulate a key that IS in CONFIG_PATHS but its path doesn't fully exist in the loaded JSON
        CONFIG_PATHS["MISSING_PATH_KEY"] = ["non_existent_section", "some_param"]
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act & Assert
            with self.assertRaises(KeyError):
                config_singleton.get("MISSING_PATH_KEY")
        del CONFIG_PATHS["MISSING_PATH_KEY"]

    def test_set_existing_key(self):
        # Arrange
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))) as mocked_file:
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            new_value = "gemma"
            # Act
            config_singleton.set(PIPELINE_TAGGING, new_value)
            retrieved_value = config_singleton.get(PIPELINE_TAGGING)
            # Assert
            self.assertEqual(retrieved_value, new_value)
            self.assertEqual(config_singleton.config["pipeline"]["tagging"], new_value)

    def test_set_non_existent_key_in_config_paths(self):
        # Arrange
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act & Assert
            with self.assertRaisesRegex(KeyError, "Key 'NON_EXISTENT_KEY_SET' not found in configuration."):
                config_singleton.set("NON_EXISTENT_KEY_SET", "some_value")

    def test_set_key_path_not_in_file(self):
        # Arrange
        CONFIG_PATHS["SET_MISSING_PATH_KEY"] = ["new_section", "new_param"]
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_config_data))):
            config_singleton = ConfigSingleton(config_file='mock_config.json')
            # Act & Assert
            with self.assertRaises(KeyError):
                config_singleton.set("SET_MISSING_PATH_KEY", "new_value")
        del CONFIG_PATHS["SET_MISSING_PATH_KEY"]

    def test_initialization_only_once(self):
        # Arrange
        mock_json_data = json.dumps(self.mock_config_data)
        with patch('builtins.open', mock_open(read_data=mock_json_data)) as mocked_file:
            # Act
            instance1 = ConfigSingleton(config_file='mock_config.json')
            instance1.set(SAVE_INTERMEDIATE_FILES, False)
            instance2 = ConfigSingleton(config_file='another_config.json')
            
            # Assert
            mocked_file.assert_called_once_with(os.path.join(CONFIG_DIR, 'mock_config.json'), 'r')
            self.assertIs(instance1, instance2)
            self.assertEqual(instance2.get(SAVE_INTERMEDIATE_FILES), False)
            self.assertEqual(instance1.config, instance2.config)

if __name__ == '__main__':
    unittest.main()
