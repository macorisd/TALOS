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
    "ram_plus",
    "qwen",
    "gemma",
    "minicpm",
    ["qwen", "deepseek"],
    ["qwen", "qwen"],
    ["qwen", "minicpm"],
    ["qwen", "llama"],
    ["minicpm", "deepseek"],
    ["minicpm", "qwen"],
    ["minicpm", "minicpm"],
    ["minicpm", "llama"],
    ["llava", "deepseek"],
    ["llava", "qwen"],
    ["llava", "minicpm"],
    ["llava", "llama"]
]

for model in tagging_models:
    subprocess.run(
        ["python", "src/evaluation/datasets/LVIS/talos/talos_pipeline_single_model.py"] + (model if isinstance(model, list) else [model]),
        check=True
    )

    # Rename the output segmentation directory
    if isinstance(model, list):
        new_name = f"masks_{model[0]}_{model[1]}"
    else:
        new_name = f"masks_{model}"

    new_output_dir = os.path.join(os.path.dirname(OUTPUT_SEGMENTATION_DIR), new_name)

    if os.path.exists(OUTPUT_SEGMENTATION_DIR):
        os.rename(OUTPUT_SEGMENTATION_DIR, new_output_dir)
        print(f"Renamed {OUTPUT_SEGMENTATION_DIR} -> {new_output_dir}")
    else:
        raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_SEGMENTATION_DIR}")

