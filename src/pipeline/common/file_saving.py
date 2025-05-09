import time
import os


class FileSaving:
    @staticmethod
    def create_output_directory(parent_dir: str, output_name: str, alias: str) -> str:
        """
        Create a timestamped output directory for saving outputs.
        """
        # Create parent output directory if it does not exist
        os.makedirs(parent_dir, exist_ok=True)
        
        # Prepare timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Output timestamped directory path
        base_output_timestamped_dir = os.path.join(
            parent_dir,
            f"{output_name}_{timestamp}_{alias}"
        )

        # Ensure the output directory is unique
        output_timestamped_dir = base_output_timestamped_dir
        counter = 1

        while os.path.exists(output_timestamped_dir):
            output_timestamped_dir = f"{base_output_timestamped_dir}_{counter}"
            counter += 1

        # Create the unique timestamped output directory
        os.makedirs(output_timestamped_dir)

        return output_timestamped_dir
