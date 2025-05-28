import os
import json
import argparse

from pipeline.pipeline_main import PipelineTALOS

# Read tagging model(s) from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("tagging_model", type=str, nargs="+", help="Tagging model(s) for the pipeline.")

args = parser.parse_args()
tagging_model = args.tagging_model

# Initialize the pipeline
pipeline = PipelineTALOS()

# Set the tagging strategy based on the provided argument
if len(tagging_model) == 1:
    pipeline.set_tagging_strategy(tagging_model[0])
elif len(tagging_model) == 2:
    pipeline.set_tagging_strategy(tagging_model)
else:
    raise ValueError("Invalid number of tagging models provided. Please provide one or two models.")

# Get LVIS image names
script_dir = os.path.dirname(os.path.abspath(__file__))

lvis_detections_file = os.path.join(
    script_dir,
    "..",
    "data",
    "lvis_detections.json"
)

with open(lvis_detections_file, 'r') as file:
    lvis_detections = json.load(file)

lvis_image_names = [image["image_name"] for image in lvis_detections[:3]]

# Run the pipeline
total_time, average_time = pipeline.run(lvis_image_names)

# Write the results to a txt file
output_file = os.path.join(script_dir, "masks_" + "_".join(tagging_model) + ".txt")
with open(output_file, 'w') as file:
    file.write(f"PIPELINE: {tagging_model}\n")
    file.write(f"Total execution time: {total_time:.2f} seconds\n")
    if average_time is not None:
        file.write(f"Average execution time: {average_time:.2f} seconds\n")
