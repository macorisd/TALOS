import os
import json
import numpy as np
from typing import List, Dict, Union, Tuple
from skimage.transform import resize

script_dir = os.path.dirname(os.path.abspath(__file__))

STR_PREFIX = "\n[EVALUATION | LVIS]"

# Paths

lvis_data_dir = os.path.join(
    script_dir,
    "data"
)

lvis_detections_file = os.path.join(
    lvis_data_dir,
    "lvis_detections.json"
)

lvis_masks_dir = os.path.join(
    lvis_data_dir,
    "lvis_masks"
)

talos_data_dir = os.path.join(
    script_dir,
    "talos"
)

talos_detections_dir = os.path.join(
    talos_data_dir,
    "talos_masks"
)

# List of TALOS detections subdirectories (one per image)
talos_detections_subdirs = sorted([
    name for name in os.listdir(talos_detections_dir)
    if os.path.isdir(os.path.join(talos_detections_dir, name))
])

output_results_dir = os.path.join(
    script_dir,
    "..",
    "output"
)

os.makedirs(output_results_dir, exist_ok=True)


# Check if the number of LVIS detections is equal to the number of TALOS detections

with open(lvis_detections_file, 'r') as file:
    lvis_images = json.load(file)

lvis_image_count = len(lvis_images)

talos_image_count = len(talos_detections_subdirs)

if lvis_image_count != talos_image_count:
    print(f"{STR_PREFIX} Number of LVIS images: {lvis_image_count}")
    print(f"{STR_PREFIX} Number of TALOS images: {talos_image_count}")
    raise ValueError("The number of LVIS images is not equal to the number of TALOS images.")

print(f"{STR_PREFIX} Number of images: {lvis_image_count}")


# Function to build final metrics dictionary

def add_image_metrics(
        metrics: List[Dict[str, Union[int, float]]],
        detection_count_score: int,
        label_coincidence_precision: float,
        label_coincidence_recall: float,
        bbox_similarity_score: float,
        mask_similarity_score: float,
        execution_time: float = None
) -> List[Dict[str, Union[int, float]]]:
    metrics.append({
        "detection_count_score": detection_count_score,
        "label_coincidence_precision": label_coincidence_precision,
        "label_coincidence_recall": label_coincidence_recall,
        "bbox_similarity_score": bbox_similarity_score,
        "mask_similarity_score": mask_similarity_score,
        "execution_time": execution_time
    })

    return metrics


# Functions to calculate metrics

def calculate_detection_count_score(lvis_detection_count, talos_detection_count) -> int | None:
    """
    Calculate the detection count score based on the difference between LVIS and TALOS detection counts.
    The score ranges from 0 to 100.
    - f(0) = 100
    - f(15) = 5
    - f(x) | x > 15 = 0.
    """
    if lvis_detection_count == 0 or talos_detection_count == 0:
        return None
    
    difference = abs(lvis_detection_count - talos_detection_count)
    return max(0, (-95 / 15 * difference) + 100)

def calculate_label_coincidence_precision(
    lvis_labels: List[List[str]],
    talos_labels: List[str]
) -> Tuple[float | None, List[Tuple[str, str]]]:
    """
    Calculate the label coincidence precision based on the number of detected TALOS labels that are present in the LVIS dataset.
    The score ranges from 0 to 100.
    """
    if not lvis_labels or not talos_labels:
        return None, []

    talos_labels_set = set(talos_labels)
    talos_label_count = len(talos_labels_set)
    lvis_coincidence_count = 0
    coincidences = []

    for talos_label in talos_labels_set:
        matched = False
        for lvis_label_synonyms in lvis_labels:
            for synonym in lvis_label_synonyms:
                if synonym in talos_label or talos_label in synonym:
                    lvis_coincidence_count += 1
                    coincidences.append((synonym, talos_label))
                    matched = True
                    break
            if matched:
                break

    score = (lvis_coincidence_count / talos_label_count) * 100
    return score, coincidences

def calculate_label_coincidence_recall(
    lvis_labels: List[List[str]],
    talos_labels: List[str]
) -> Tuple[float | None, List[Tuple[str, str]]]:
    """
    Calculate the label coincidence recall based on the number of LVIS labels that are present in TALOS detections.
    The score ranges from 0 to 100.
    """
    if not lvis_labels or not talos_labels:
        return None, []

    talos_labels_set = set(talos_labels)

    lvis_label_count = len(lvis_labels)
    talos_coincidence_count = 0
    coincidences = []

    for lvis_label_synonyms in lvis_labels:
        matched = False
        for synonym in lvis_label_synonyms:
            for talos_label in talos_labels_set:
                if synonym in talos_label or talos_label in synonym:
                    talos_coincidence_count += 1
                    coincidences.append((synonym, talos_label))
                    matched = True
                    break
            if matched:
                break

    score = (talos_coincidence_count / lvis_label_count) * 100
    return score, coincidences

def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def calculate_bbox_similarity_score(
        lvis_detections: List[Dict],
        talos_detections: List[Dict]
) -> Tuple[float | None, List[Tuple[int, int]]]:
    """
    Calculate the bbox similarity score based on the bbox IoU between LVIS and TALOS detections with label coincidence.
    The score ranges from 0 to 100.
    """

    if not lvis_detections or not talos_detections:
        return None, []

    iou_scores = []
    bbox_coincidence_ids = []

    for lvis_det in lvis_detections:
        lvis_id = lvis_det["id"]
        lvis_bbox = lvis_det["bbox"]
        lvis_labels = lvis_det["labels"]

        best_iou = 0.0
        best_talos_id = None

        for talos_det in talos_detections:
            talos_id = talos_det["id"]

            talos_label = talos_det["label"]
            talos_bbox = talos_det["bbox"]

            # Check label match
            label_match = any(
                synonym in talos_label or talos_label in synonym
                for synonym in lvis_labels
            )
            if not label_match:
                continue

            # Compute IoU
            iou = calculate_bbox_iou(lvis_bbox, talos_bbox)
            if iou > best_iou:
                best_iou = iou
                best_talos_id = talos_id

        if best_talos_id is not None and best_iou > 0:
            iou_scores.append(best_iou * 100)
            bbox_coincidence_ids.append((lvis_id, best_talos_id))

    if not iou_scores:
        return 0.0, []

    avg_score = sum(iou_scores) / len(iou_scores)
    return avg_score, bbox_coincidence_ids

def calculate_mask_iou(mask1: List[List[int]], mask2: List[List[int]]) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.
    """
    intersection = sum(1 for i in range(len(mask1)) for j in range(len(mask1[0])) if mask1[i][j] and mask2[i][j])
    area1 = sum(1 for row in mask1 for pixel in row if pixel)
    area2 = sum(1 for row in mask2 for pixel in row if pixel)
    union = area1 + area2 - intersection

    print(f"Intersection: {intersection}, Area1: {area1}, Area2: {area2}, Union: {union}")

    if union == 0:
        return 0.0

    return intersection / union

def calculate_mask_similarity_score(
        lvis_detections: List[Dict],
        talos_detections: List[Dict],
        bbox_coincidence_ids: List[Tuple[int, int]],
        image_name: str,
        talos_subdir: str,
        talos_segmentation_alias: str = "sam2"
) -> Tuple[float | None]:
    """
    Calculate the mask similarity score based on the mask IoU between LVIS and TALOS detections with similar bounding boxes.
    The score ranges from 0 to 100.
    """

    if not lvis_detections or not talos_detections or not bbox_coincidence_ids:
        return None

    mask_scores = []
    mask_coincidence_ids = []

    for lvis_id, talos_id in bbox_coincidence_ids:
        lvis_det = next((d for d in lvis_detections if d["id"] == lvis_id), None)
        talos_det = next((d for d in talos_detections if d["id"] == talos_id), None)

        if lvis_det is None or talos_det is None:
            continue

        lvis_mask_file = os.path.join(
            lvis_masks_dir,
            f"{image_name}_mask_{lvis_id}.npz"
        )

        talos_mask_file = os.path.join(
            talos_detections_dir,
            talos_subdir,
            f"segmentation_{talos_segmentation_alias}_mask_{talos_id}.npz"
        )

        # Load masks (assuming they are binary masks)
        lvis_mask = np.load(lvis_mask_file)["mask"]
        talos_mask = np.load(talos_mask_file)["mask"]

        # Resize talos_mask to 256x256
        talos_mask_resized = resize(
            talos_mask,
            lvis_mask.shape,  # (256, 256)
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)

        # Compute IoU
        iou = calculate_mask_iou(lvis_mask, talos_mask_resized)
        mask_scores.append(iou * 100)
        mask_coincidence_ids.append((lvis_id, talos_id))

    if not mask_scores:
        return 0.0, []

    avg_score = sum(mask_scores) / len(mask_scores)
    return avg_score


# Iterate through LVIS detections and evaluate TALOS detections

metrics = []

for i, lvis_image in enumerate(lvis_images):
    print("\n----------------------------------------------")
    print(f"{STR_PREFIX} Evaluating image {i+1} ({lvis_image['image_name']})...")

    # Metrics initialization

    detection_count_score = None
    label_coincidence_precision = None
    label_coincidence_recall = None
    bbox_similarity_score = None
    mask_similarity_score = None
    execution_time = None

    # Load TALOS detections for the current image

    talos_image_file = os.path.join(
        talos_detections_dir,
        talos_detections_subdirs[i],
        "detections_info.json"
    )

    with open(talos_image_file, 'r') as file:
        talos_image = json.load(file)
    
    # Evaluate detection count

    lvis_detection_count = len(lvis_image["detections"])
    talos_detection_count = len(talos_image["detections"])

    detection_count_score = calculate_detection_count_score(lvis_detection_count, talos_detection_count)

    print(f"{STR_PREFIX} LVIS detection count for image {i+1}: {lvis_detection_count}")
    print(f"{STR_PREFIX} TALOS detection count for image {i+1}: {talos_detection_count}")
    print(f"{STR_PREFIX} Detection count score for image {i+1}: {detection_count_score}")

    # Evaluate label coincidence precision (LVIS labels that are present in TALOS detections)

    lvis_labels = [detection["labels"] for detection in lvis_image["detections"]]
    talos_labels = [detection["label"] for detection in talos_image["detections"]]

    print(f"{STR_PREFIX} LVIS labels for image {i+1}: {lvis_labels}")
    print(f"{STR_PREFIX} TALOS labels for image {i+1}: {talos_labels}")

    label_coincidence_precision, precision_label_coincidences = calculate_label_coincidence_precision(lvis_labels, talos_labels)

    print(f"{STR_PREFIX} LVIS label coincidences in TALOS (precision) for image {i+1}: {precision_label_coincidences}")
    print(f"{STR_PREFIX} LVIS label coincidence score (precision) for image {i+1}: {label_coincidence_precision}")

    # Evaluate label coincidence recall (TALOS labels that are present in LVIS ground truth)

    label_coincidence_recall, recall_label_coincidences = calculate_label_coincidence_recall(lvis_labels, talos_labels)

    print(f"{STR_PREFIX} TALOS label coincidences in LVIS (recall) for image {i+1}: {recall_label_coincidences}")
    print(f"{STR_PREFIX} TALOS label coincidence score (recall) for image {i+1}: {label_coincidence_recall}")

    # Evaluate bbox similarity for the detections with label coincidence

    precision_label_coincidence_detections = []
    recall_label_coincidence_detections = []

    for j, detection in enumerate(lvis_image["detections"]):
        matched = False
        for synonym in detection["labels"]:
            for lvis_label_coincidence in precision_label_coincidences:
                if synonym in lvis_label_coincidence[0] or lvis_label_coincidence[0] in synonym:
                    precision_label_coincidence_detections.append(detection)
                    matched = True
                    break
            
            if matched:
                break
    
    for j, detection in enumerate(talos_image["detections"]):
        for talos_label_coincidence in recall_label_coincidences:
            if detection["label"] in talos_label_coincidence[1] or talos_label_coincidence[1] in detection["label"]:
                recall_label_coincidence_detections.append(detection)
                break
    
    bbox_similarity_score, bbox_coincidence_ids = calculate_bbox_similarity_score(
        precision_label_coincidence_detections,
        recall_label_coincidence_detections
    )

    print(f"{STR_PREFIX} Bbox coincidences for image {i+1}: {bbox_coincidence_ids}")
    print(f"{STR_PREFIX} Bbox similarity score for image {i+1}: {bbox_similarity_score}")

    # Evaluate mask similarity for the detections with label coincidence

    mask_similarity_score = calculate_mask_similarity_score(
        precision_label_coincidence_detections,
        recall_label_coincidence_detections,
        bbox_coincidence_ids,
        lvis_image["image_name"],
        talos_detections_subdirs[i]
    )

    print(f"{STR_PREFIX} Mask similarity score for image {i+1}: {mask_similarity_score}")

    # Add metrics to the list
    metrics = add_image_metrics(
        metrics,
        detection_count_score,
        label_coincidence_precision,
        label_coincidence_recall,
        bbox_similarity_score,
        mask_similarity_score
    )

    print(f"{STR_PREFIX} Metrics for image {i+1}: {metrics[-1]}")
    print(f"{STR_PREFIX} Finished evaluating image {i+1}.\n")


# Calculate average scores

avg_detection_count_score = np.mean([m["detection_count_score"] for m in metrics if m["detection_count_score"] is not None])
avg_label_coincidence_precision = np.mean([m["label_coincidence_precision"] for m in metrics if m["label_coincidence_precision"] is not None])
avg_label_coincidence_recall = np.mean([m["label_coincidence_recall"] for m in metrics if m["label_coincidence_recall"] is not None])
avg_bbox_similarity_score = np.mean([m["bbox_similarity_score"] for m in metrics if m["bbox_similarity_score"] is not None])
avg_mask_similarity_score = np.mean([m["mask_similarity_score"] for m in metrics if m["mask_similarity_score"] is not None])

# Print final metrics

print("\n----------------------------------------------")

print(f"{STR_PREFIX} Final metrics:")
print(f"{STR_PREFIX} Average detection count score: {avg_detection_count_score}")
print(f"{STR_PREFIX} Average label coincidence precision score: {avg_label_coincidence_precision}")
print(f"{STR_PREFIX} Average label coincidence recall score: {avg_label_coincidence_recall}")
print(f"{STR_PREFIX} Average bbox similarity score: {avg_bbox_similarity_score}")
print(f"{STR_PREFIX} Average mask similarity score: {avg_mask_similarity_score}")

average_final_score = (
    avg_detection_count_score +
    avg_label_coincidence_precision +
    avg_label_coincidence_recall +
    avg_bbox_similarity_score +
    avg_mask_similarity_score
) / 5

print(f"{STR_PREFIX} Average final score: {average_final_score}")

# Save metrics summary to file

output_file = os.path.join(
    output_results_dir,
    "talos_lvis_evaluation.json"
)

metrics_summary = {
    "average_detection_count_score": avg_detection_count_score,
    "average_label_coincidence_precision_score": avg_label_coincidence_precision,
    "average_label_coincidence_recall_score": avg_label_coincidence_recall,
    "average_bbox_similarity_score": avg_bbox_similarity_score,
    "average_mask_similarity_score": avg_mask_similarity_score,
    "average_final_score": average_final_score
}

with open(output_file, 'w') as file:
    json.dump(metrics_summary, file, indent=4)

print(f"{STR_PREFIX} Metrics saved to {output_file}")
