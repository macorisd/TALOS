# LVIS dataset formatting script.
# https://www.lvisdataset.org/dataset
# Last downloaded in March 2025, please check if the format is still the same.

"""
LVIS Dataset Formatter and Mask Generator
------------------------------------------
This script performs two main tasks:
1. Formats the original LVIS-style JSON file.
2. Generates binary masks from polygon segmentations and saves them as .npz and optionally .png.
"""

import os
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

# Configuration
MASK_SIZE = (256, 256)
SAVE_IMAGES = True
MAX_IMAGES = 1000  # Number of images to process from the dataset
SAVE_INTERMEDIATE_JSON = False

# Setup paths
script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')
input_json_path = os.path.join(script_directory, 'raw_data.json')
formatted_json_path = os.path.join(script_directory, 'formatted_data.json')
output_dir = os.path.join(script_directory, 'output_masks')
os.makedirs(output_dir, exist_ok=True)

# Step 1: Format the LVIS dataset
print("Loading LVIS JSON file...", end=" ", flush=True)
with open(input_json_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
print("Done.")

# Map image_id to (width, height)
image_dimensions = {img["id"]: (img["width"], img["height"]) for img in raw_data.get("images", [])}

# Prepare category map with labels
print("Processing categories...", end=" ", flush=True)
category_map = {}
for category in raw_data.get('categories', []):
    name = category.get('name', '')
    synonyms = category.get('synonyms', [])
    labels = list(set([name] + synonyms))
    category['labels'] = labels
    category_map[category['id']] = labels
    category.pop('name', None)
    category.pop('synonyms', None)
print("Done.")

# Build formatted dataset
print("Formatting annotations...", end=" ", flush=True)
image_detections = defaultdict(list)
image_metadata = {}

for annotation in raw_data.get('annotations', []):
    image_id = annotation.pop('image_id')
    image_name = f"{image_id}.jpg"

    if image_name not in image_metadata:
        width, height = image_dimensions.get(image_id, (None, None))
        image_metadata[image_name] = {"width": width, "height": height}

    detection = {
        "id": len(image_detections[image_name]) + 1,
        "labels": category_map.get(annotation.pop('category_id'), []),
        "bbox": annotation.pop("bbox", []),
        "segmentation": annotation.pop("segmentation", [])
    }
    image_detections[image_name].append(detection)

formatted_data = [
    {
        "image_name": image_name,
        "width": image_metadata[image_name]["width"],
        "height": image_metadata[image_name]["height"],
        "detections": detections
    }
    for image_name, detections in list(image_detections.items())[:MAX_IMAGES]
]
print("Done.")

if SAVE_INTERMEDIATE_JSON:
    # Save formatted JSON
    print("Saving formatted JSON...", end=" ", flush=True)
    with open(formatted_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False)
    print("Done.")

# Step 2: Generate binary masks
print("Generating binary masks...", flush=True)

def polygons_to_binary_mask(segmentation, image_size, mask_size=MASK_SIZE):
    orig_width, orig_height = image_size
    mask_width, mask_height = mask_size

    def scale_coords(coords):
        return [(x * mask_width / orig_width, y * mask_height / orig_height) for x, y in coords]

    mask = Image.new('1', mask_size, 0)
    draw = ImageDraw.Draw(mask)

    for polygon in segmentation:
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        scaled_coords = scale_coords(coords)

        if Polygon(scaled_coords).is_valid:
            draw.polygon(scaled_coords, outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)

for image_info in formatted_data:
    image_name = image_info['image_name']
    orig_size = (image_info['width'], image_info['height'])

    for detection in image_info['detections']:
        segmentation = detection.get('segmentation')
        detection_id = detection['id']

        if segmentation:
            binary_mask = polygons_to_binary_mask(segmentation, orig_size)

            # Save mask as .npz
            npz_filename = f"{image_name}_mask_{detection_id}.npz"
            np.savez_compressed(os.path.join(output_dir, npz_filename), mask=binary_mask)

            # Optionally save as PNG
            if SAVE_IMAGES:
                png_image = Image.fromarray(binary_mask * 255)
                png_filename = f"{image_name}_mask_{detection_id}.png"
                png_image.save(os.path.join(output_dir, png_filename))

            # Remove segmentation after processing
            detection.pop('segmentation', None)

# Save final JSON without segmentations
final_json_path = os.path.join(script_directory, 'final_data.json')
with open(final_json_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2)

print("All masks saved and final JSON created.")
print(f"Formatted JSON: {formatted_json_path}")
print(f"Final JSON without segmentations: {final_json_path}")
