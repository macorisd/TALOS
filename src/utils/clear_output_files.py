import os
import shutil
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent

directories = [
    base_path / "pipeline" / "output" / "tagging_output",
    base_path / "pipeline" / "output" / "location_output",
    base_path / "pipeline" / "output" / "segmentation_output"
]

for directory in directories:
    for filename in os.listdir(directory):
        file_path = directory / filename

        if filename.endswith(".md"):
            continue

        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print("All output files have been deleted.")
