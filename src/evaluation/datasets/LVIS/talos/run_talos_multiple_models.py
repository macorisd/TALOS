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

lvis_image_names = [image["image_name"] for image in lvis_detections]

tagging_models = [
    # "ram_plus",                       # DONE
    # "qwen",                           # DONE
    # "gemma",                          # DONE
    # "minicpm",                        # DONE
    # ["qwen", "deepseek"],             # 800/1000
    # ["qwen", "qwen"],                 #
    ["qwen", "minicpm"],              #
    # ["qwen", "llama"],                #
    ["minicpm", "deepseek"],          #
    # ["minicpm", "qwen"],              #
    # ["minicpm", "minicpm"],           # DONE
    # ["minicpm", "llama"],             #
    ["llava", "deepseek"],            #
    # ["llava", "qwen"],                #
    # ["llava", "minicpm"],             # DONE
    # ["llava", "llama"]                #
]

for model in tagging_models:
    subprocess.run(
        ["python", "src/evaluation/datasets/LVIS/talos/run_talos_single_model.py"]
        + (model if isinstance(model, list) else [model]),
        check=True
    )

    # Determine new base name
    if isinstance(model, list):
        base_name = f"masks_{model[0]}_{model[1]}"
    else:
        base_name = f"masks_{model}"

    new_output_dir = os.path.join(os.path.dirname(OUTPUT_SEGMENTATION_DIR), base_name)

    # Check if directory already exists and add suffix if needed
    final_output_dir = new_output_dir
    suffix = 1
    while os.path.exists(final_output_dir):
        final_output_dir = f"{new_output_dir}_{suffix}"
        suffix += 1

    if os.path.exists(OUTPUT_SEGMENTATION_DIR):
        os.rename(OUTPUT_SEGMENTATION_DIR, final_output_dir)
        print(f"\nRenamed {OUTPUT_SEGMENTATION_DIR} -> {final_output_dir}")
    else:
        raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_SEGMENTATION_DIR}")
