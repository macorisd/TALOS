import os
import json
import subprocess

from pipeline.config.paths import OUTPUT_SEGMENTATION_DIR

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

tagging_models = [
    "qwen",
    "gemma",
    ["llava", "deepseek"],
    ["qwen", "deepseek"],
    "ram_plus"
]


for model in tagging_models:
    subprocess.run(
        ["python", "src/evaluation/datasets/LVIS/talos/run_talos_pipeline.py"] + (model if isinstance(model, list) else [model]),
        check=True
    )

    # Rename the output segmentation directory
    if isinstance(model, list):
        new_name = f"masks_{model[0]}_{model[1]}"
    else:
        new_name = f"masks_{model}"

    output_dir = OUTPUT_SEGMENTATION_DIR
    new_output_dir = os.path.join(os.path.dirname(output_dir), new_name)

    if os.path.exists(output_dir):
        os.rename(output_dir, new_output_dir)
        print(f"Renamed {output_dir} -> {new_output_dir}")
    else:
        print(f"Warning: {output_dir} no existe, no se pudo renombrar.")

