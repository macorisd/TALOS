import os
import json

# Get the path to the script directory

script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')

print("Reading file...", end=" ", flush=True)

# Load the JSON file

json_path = os.path.join(script_directory, 'lvis_v1_val.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("Done", flush=True)

# Remove useless general keys

print("Removing 'info' and 'licenses' keys...", end=" ", flush=True)
data.pop('info', None)
data.pop('licenses', None)
print("Done", flush=True)

# Remove specific keys from the 'images' list

print("Cleaning image dictionaries...", end=" ", flush=True)

image_keys_to_remove = [
    'date_captured', 'neg_category_ids', 'license', 
    'flickr_url', 'coco_url', 'not_exhaustive_category_ids'
]

for image in data.get('images', []):
    for key in image_keys_to_remove:
        image.pop(key, None)
print("Done", flush=True)

# Remove specific keys from the 'categories' list

print("Cleaning category dictionaries...", end=" ", flush=True)
category_keys_to_remove = [
    'image_count', 'def', 'synset', 'frequency', 'instance_count'
]

for category in data.get('categories', []):
    for key in category_keys_to_remove:
        category.pop(key, None)
print("Done", flush=True)

# Check if each 'name' is present in 'synonyms' for each category in 'categories'

print("Checking if each 'name' is present in 'synonyms'...", end=" ", flush=True)
for idx, category in enumerate(data.get('categories', []), start=1):
    name = category.get('name', '')
    synonyms = category.get('synonyms', [])
    if name not in synonyms:
        print(f"Warning: Category at index {idx} with name '{name}' not found in its synonyms list.", flush=True)
print("Done", flush=True)

# Save the filtered JSON

print("Saving filtered JSON...", end=" ", flush=True)

output_path = os.path.join(script_directory, 'lvis_v1_val_filtered.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
print("Done", flush=True)

print(f"Filtered JSON saved to: {output_path}", flush=True)