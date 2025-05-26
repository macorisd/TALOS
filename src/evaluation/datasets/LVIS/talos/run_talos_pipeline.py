import os
import json
import argparse

from pipeline.pipeline_main import PipelineTALOS


# Tagging method argument

parser = argparse.ArgumentParser()
parser.add_argument("tagging_model", type=str, nargs="+", help="Tagging model(s) for the pipeline.")

args = parser.parse_args()

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

# Run the pipeline with the specified tagging strategy

pipeline = PipelineTALOS()

# Set the tagging strategy based on the provided argument
if len(args.tagging_model) == 1:
    pipeline.set_tagging_strategy(args.tagging_model[0])
elif len(args.tagging_model) == 2:
    pipeline.set_tagging_strategy(args.tagging_model)
else:
    raise ValueError("Invalid number of tagging models provided. Please provide one or two models.")

pipeline.run(lvis_image_names)
