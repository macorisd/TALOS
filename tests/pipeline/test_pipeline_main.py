import unittest
from unittest.mock import patch, MagicMock, call
import time

from pipeline.pipeline_main import PipelineTALOS, main
from pipeline.config.config import (
    PIPELINE_TAGGING,
    PIPELINE_LOCATION,
    PIPELINE_SEGMENTATION
)

class TestPipelineTALOS(unittest.TestCase):

    @patch('pipeline.pipeline_main.StrategyFactory')
    @patch('pipeline.pipeline_main.ConfigSingleton')
    @patch('pipeline.pipeline_main.print_purple')
    def test_initialization_loads_config_and_sets_strategies(self, mock_print_purple, MockConfigSingleton, MockStrategyFactory):
        # Arrange
        mock_config_instance = MockConfigSingleton.return_value
        mock_config_instance.get.side_effect = lambda key: {
            PIPELINE_TAGGING: "mock_tagging_method",
            PIPELINE_LOCATION: "mock_location_method",
            PIPELINE_SEGMENTATION: "mock_segmentation_method"
        }[key]

        mock_tagging_strategy = MagicMock()
        mock_location_strategy = MagicMock()
        mock_segmentation_strategy = MagicMock()

        MockStrategyFactory.create_tagging_strategy.return_value = mock_tagging_strategy
        MockStrategyFactory.create_location_strategy.return_value = mock_location_strategy
        MockStrategyFactory.create_segmentation_strategy.return_value = mock_segmentation_strategy

        # Act
        pipeline = PipelineTALOS()

        # Assert
        MockConfigSingleton.assert_called_once()
        mock_config_instance.get.assert_any_call(PIPELINE_TAGGING)
        mock_config_instance.get.assert_any_call(PIPELINE_LOCATION)
        mock_config_instance.get.assert_any_call(PIPELINE_SEGMENTATION)

        MockStrategyFactory.create_tagging_strategy.assert_called_once_with("mock_tagging_method")
        MockStrategyFactory.create_location_strategy.assert_called_once_with("mock_location_method")
        MockStrategyFactory.create_segmentation_strategy.assert_called_once_with("mock_segmentation_method")

        self.assertEqual(pipeline.tagging_strategy, mock_tagging_strategy)
        self.assertEqual(pipeline.location_strategy, mock_location_strategy)
        self.assertEqual(pipeline.segmentation_strategy, mock_segmentation_strategy)
        mock_print_purple.assert_called()

    @patch('pipeline.pipeline_main.time.time')
    @patch('pipeline.pipeline_main.StrategyFactory')
    @patch('pipeline.pipeline_main.ConfigSingleton')
    @patch('pipeline.pipeline_main.print_purple')
    def test_run_single_image_single_iteration(self, mock_print_purple, MockConfigSingleton, MockStrategyFactory, mock_time):
        # Arrange
        mock_config_instance = MockConfigSingleton.return_value
        mock_config_instance.get.side_effect = lambda key: {
            PIPELINE_TAGGING: "tag_method",
            PIPELINE_LOCATION: "loc_method",
            PIPELINE_SEGMENTATION: "seg_method"
        }[key]

        mock_tagging_strategy = MagicMock()
        mock_location_strategy = MagicMock()
        mock_segmentation_strategy = MagicMock()

        MockStrategyFactory.create_tagging_strategy.return_value = mock_tagging_strategy
        MockStrategyFactory.create_location_strategy.return_value = mock_location_strategy
        MockStrategyFactory.create_segmentation_strategy.return_value = mock_segmentation_strategy

        # Mock return values for strategy execute methods
        mock_tags = ["tag1", "tag2"]
        mock_locations = [((0,0,10,10), "tag1")]
        mock_segmentation_info = {"info": "data"}
        mock_all_masks = [MagicMock()] # list of mask objects

        mock_tagging_strategy.execute.return_value = mock_tags
        mock_location_strategy.execute.return_value = mock_locations
        mock_segmentation_strategy.execute.return_value = (mock_segmentation_info, mock_all_masks)

        # Mock time.time to control duration calculation
        mock_time.side_effect = [100.0, 105.0] # start_time, end_time

        pipeline = PipelineTALOS()
        image_name = "test_image.jpg"

        # Act
        total_time, average_time = pipeline.run(input_image_names=image_name, iters=1)

        # Assert
        mock_tagging_strategy.load_inputs.assert_called_once_with(input_image_name=image_name)
        mock_tagging_strategy.execute.assert_called_once()
        mock_tagging_strategy.save_outputs.assert_called_once_with(mock_tags)

        mock_location_strategy.load_inputs.assert_called_once_with(input_image_name=image_name, input_tags=mock_tags)
        mock_location_strategy.execute.assert_called_once()
        mock_location_strategy.save_outputs.assert_called_once_with(mock_locations)

        mock_segmentation_strategy.load_inputs.assert_called_once_with(input_image_name=image_name, input_location=mock_locations)
        mock_segmentation_strategy.execute.assert_called_once()
        mock_segmentation_strategy.save_outputs.assert_called_once_with(mock_segmentation_info, mock_all_masks)

        self.assertAlmostEqual(total_time, 5.0)
        self.assertIsNone(average_time)
        mock_print_purple.assert_called()

    @patch('pipeline.pipeline_main.time.time')
    @patch('pipeline.pipeline_main.StrategyFactory')
    @patch('pipeline.pipeline_main.ConfigSingleton')
    @patch('pipeline.pipeline_main.print_purple')
    def test_run_multiple_images_multiple_iterations(self, mock_print_purple, MockConfigSingleton, MockStrategyFactory, mock_time):
        # Arrange
        mock_config_instance = MockConfigSingleton.return_value
        mock_config_instance.get.side_effect = lambda key: {
            PIPELINE_TAGGING: "tag_method",
            PIPELINE_LOCATION: "loc_method",
            PIPELINE_SEGMENTATION: "seg_method"
        }[key]

        mock_tagging_strategy = MagicMock()
        mock_location_strategy = MagicMock()
        mock_segmentation_strategy = MagicMock()

        MockStrategyFactory.create_tagging_strategy.return_value = mock_tagging_strategy
        MockStrategyFactory.create_location_strategy.return_value = mock_location_strategy
        MockStrategyFactory.create_segmentation_strategy.return_value = mock_segmentation_strategy

        mock_tags = ["tag1"]
        mock_locations = [((0,0,5,5), "tag1")]
        mock_segmentation_info = {"s_info": "s_data"}
        mock_all_masks = [MagicMock()]

        mock_tagging_strategy.execute.return_value = mock_tags
        mock_location_strategy.execute.return_value = mock_locations
        mock_segmentation_strategy.execute.return_value = (mock_segmentation_info, mock_all_masks)

        mock_time.side_effect = [100, 102, 104, 106, 108, 110, 112, 114]

        pipeline = PipelineTALOS()
        image_names = ["image1.png", "image2.png"]
        iters = 2

        # Act
        total_time, average_time = pipeline.run(input_image_names=image_names, iters=iters)

        # Assert
        self.assertEqual(mock_tagging_strategy.load_inputs.call_count, 4)
        self.assertEqual(mock_tagging_strategy.execute.call_count, 4)
        self.assertEqual(mock_tagging_strategy.save_outputs.call_count, 4)

        self.assertEqual(mock_location_strategy.load_inputs.call_count, 4)
        self.assertEqual(mock_location_strategy.execute.call_count, 4)
        self.assertEqual(mock_location_strategy.save_outputs.call_count, 4)

        self.assertEqual(mock_segmentation_strategy.load_inputs.call_count, 4)
        self.assertEqual(mock_segmentation_strategy.execute.call_count, 4)
        self.assertEqual(mock_segmentation_strategy.save_outputs.call_count, 4)

        # Check calls for one specific image to ensure correct arguments
        expected_calls_tag_load = [
            call(input_image_name='image1.png'), call(input_image_name='image2.png'),
            call(input_image_name='image1.png'), call(input_image_name='image2.png')
        ]
        mock_tagging_strategy.load_inputs.assert_has_calls(expected_calls_tag_load, any_order=False)
        mock_tagging_strategy.save_outputs.assert_has_calls([call(mock_tags)] * 4, any_order=False)
        
        expected_calls_loc_load = [
            call(input_image_name='image1.png', input_tags=mock_tags), call(input_image_name='image2.png', input_tags=mock_tags),
            call(input_image_name='image1.png', input_tags=mock_tags), call(input_image_name='image2.png', input_tags=mock_tags)
        ]
        mock_location_strategy.load_inputs.assert_has_calls(expected_calls_loc_load, any_order=False)
        mock_location_strategy.save_outputs.assert_has_calls([call(mock_locations)] * 4, any_order=False)

        expected_calls_seg_load = [
            call(input_image_name='image1.png', input_location=mock_locations), call(input_image_name='image2.png', input_location=mock_locations),
            call(input_image_name='image1.png', input_location=mock_locations), call(input_image_name='image2.png', input_location=mock_locations)
        ]
        mock_segmentation_strategy.load_inputs.assert_has_calls(expected_calls_seg_load, any_order=False)
        mock_segmentation_strategy.save_outputs.assert_has_calls([call(mock_segmentation_info, mock_all_masks)] * 4, any_order=False)

        self.assertAlmostEqual(total_time, 8.0) # 4 runs * 2 seconds/run
        self.assertAlmostEqual(average_time, 2.0)
        mock_print_purple.assert_called()

    @patch('pipeline.pipeline_main.PipelineTALOS')
    @patch('pipeline.pipeline_main.print_purple')
    def test_main_function_success(self, mock_print_purple, MockPipelineTALOS):
        # Arrange
        mock_pipeline_instance = MockPipelineTALOS.return_value
        image_names = ["img1.jpg", "img2.jpg"]
        iters_count = 3

        # Act
        main(input_image_names=image_names, iters=iters_count)

        # Assert
        MockPipelineTALOS.assert_called_once_with()
        mock_pipeline_instance.run.assert_called_once_with(image_names, iters=iters_count)

    @patch('pipeline.pipeline_main.print_purple')
    def test_main_function_no_input_images_raises_value_error(self, mock_print_purple):
        # Arrange
        image_names = []
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            main(input_image_names=image_names, iters=1)
        self.assertIn("No input image names provided", str(context.exception))

if __name__ == '__main__':
    unittest.main()
