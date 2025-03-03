# Delete all the files in the following directories:
    # src/tagging/lvlm_llm_tagging/lvlm_description/output_descriptions
    # src/tagging/output_tags
    # src/location/output_location

import os
import shutil

directories = [
    "src/tagging/lvlm_llm_tagging/lvlm_description/output_descriptions",
    "src/tagging/output_tags",
    "src/location/output_location",
    "src/segmentation/output_segments"
]

for directory in directories:
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")