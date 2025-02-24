# Delete all the files in the following directories:
    # src/tagging/lvlm_llm_tagging/lvlm_description/output_descriptions
    # src/tagging/output_tags
    # src/location/output_location

import os

directories = [
    "src/tagging/lvlm_llm_tagging/lvlm_description/output_descriptions",
    "src/tagging/output_tags",
    "src/location/output_location"
]

for directory in directories:
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")