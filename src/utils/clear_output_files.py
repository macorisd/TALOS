import os
import shutil
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent

directories = [
    base_path / "pipeline" / "tagging" / "lvlm_llm_tagging" / "lvlm_description" / "output_descriptions",
    base_path / "pipeline" / "tagging" / "output_tags",
    base_path / "pipeline" / "location" / "output_location",
    base_path / "pipeline" / "output_segments"
]

for directory in directories:
    for filename in os.listdir(directory):
        file_path = directory / filename

        if filename == "INFO.md":
            continue

        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print("All output files have been deleted.")
