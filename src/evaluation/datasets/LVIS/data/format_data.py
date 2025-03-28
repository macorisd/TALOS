# LVIS dataset formatting script.
# https://www.lvisdataset.org/dataset
# Last downloaded in March 2025, please check if the format is still the same.

import os
import json
from collections import defaultdict

# Get the path to the script directory
script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')

print("Reading file...", end=" ", flush=True)

# Load the JSON file
json_path = os.path.join(script_directory, 'raw_data.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("Done", flush=True)

# Create a mapping from image_id to (width, height)
image_dimensions = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}

# 'categories': merge 'name' and 'synonyms' into 'labels' without duplicates
print("Merging 'name' and 'synonyms' into 'labels'...", end=" ", flush=True)
category_map = {}
for category in data.get('categories', []):
    name = category.get('name', '')
    synonyms = category.get('synonyms', [])
    labels = list(set([name] + synonyms))  # Remove duplicates
    category['labels'] = labels
    category_map[category['id']] = labels  # Store mapping of id to labels
    category.pop('name', None)  # Remove original 'name' key
    category.pop('synonyms', None)  # Remove original 'synonyms' key
print("Done", flush=True)

# Transform annotations into the new format
print("Transforming annotations...", end=" ", flush=True)
image_detections = defaultdict(list)
image_metadata = {}

for annotation in data.get('annotations', []):
    image_id = annotation.pop('image_id')
    image_name = f"{image_id}.jpg"
    
    # Store width and height for each image
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

# Build final JSON structure
formatted_data = [
    {
        "image_name": image_name,
        "width": image_metadata[image_name]["width"],
        "height": image_metadata[image_name]["height"],
        "detections": detections
    }
    for image_name, detections in image_detections.items()
]
print("Done", flush=True)

# Save the transformed JSON
print("Saving formatted JSON...", end=" ", flush=True)
output_path = os.path.join(script_directory, 'formatted_data.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data[:10000], f, ensure_ascii=False)
print("Done", flush=True)

print(f"Formatted JSON saved to: {output_path}", flush=True)
