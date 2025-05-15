import os
import json
from typing import List, Dict, Union, Tuple

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
        label_coincidence_score: float,
        bbox_similarity_score: float,
        mask_similarity_score: float,
        execution_time: float
) -> List[Dict[str, Union[int, float]]]:
    metrics.append({
        "detection_count_score": detection_count_score,
        "label_coincidence_score": label_coincidence_score,
        "bbox_similarity_score": bbox_similarity_score,
        "mask_similarity_score": mask_similarity_score,
        "execution_time": execution_time
    })

    return metrics


# Functions to calculate metrics

def calculate_detection_count_score(lvis_detection_count, talos_detection_count) -> int | None:
    """
    Calculate the detection count score based on the difference between LVIS and TALOS detection counts.
    """
    if lvis_detection_count == 0 or talos_detection_count == 0:
        return None
    
    difference = abs(lvis_detection_count - talos_detection_count)

    # f(0) = 25
    # f(15) = 1
    # f(x) | x > 15 = 0
    return max(0, (-8 / 5 * difference) + 25)

def calculate_lvis_label_coincidence_score(
    lvis_labels: List[List[str]],
    talos_labels: List[str]
) -> Tuple[float | None, List[Tuple[str, str]]]:
    """
    Calculate the label coincidence score based on the number of LVIS labels that are present in TALOS detections.
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

    score = (talos_coincidence_count / lvis_label_count) * 25
    return score, coincidences

def calculate_talos_label_coincidence_score(
    lvis_labels: List[List[str]],
    talos_labels: List[str]
) -> Tuple[float | None, List[Tuple[str, str]]]:
    """
    Calculate the label coincidence score based on the number of detected TALOS labels that are present in the LVIS dataset.
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

    score = (lvis_coincidence_count / talos_label_count) * 25
    return score, coincidences

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
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
    # TODO: Los parámetros son las detecciones (completas) filtradas por similitud de labels.
    # La función devuelve el score de similitud de bboxes (definir un threshold dinámico según el ancho)
    # y tuplas (lvis_id, talos_id) de las detecciones cuyos bboxes son similares (y comparten label).

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
            iou = calculate_iou(lvis_bbox, talos_bbox)
            if iou > best_iou:
                best_iou = iou
                best_talos_id = talos_id

        if best_talos_id is not None and best_iou > 0:
            iou_scores.append(best_iou * 20)
            bbox_coincidence_ids.append((lvis_id, best_talos_id))

    if not iou_scores:
        return 0.0, []

    avg_score = sum(iou_scores) / len(iou_scores)
    return avg_score, bbox_coincidence_ids

def calculate_mask_similarity_score(coincidence_ids: List[Tuple[int, int]]) -> float | None:
    # TODO: los parámetros son las tuplas (lvis_id, talos_id) de las detecciones cuyos bboxes son similares (y comparten label).
    # La función devuelve el score de similitud de masks (IoU).

    pass


# Iterate through LVIS detections and evaluate TALOS detections

metrics = []

for i, lvis_image in enumerate(lvis_images):
    print(f"\n{STR_PREFIX} Evaluating image {i+1}...")

    # Metrics initialization

    detection_count_score = None
    label_coincidence_score = None
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

    # Evaluate label coincidence (LVIS labels that are present in TALOS detections)

    lvis_labels = [detection["labels"] for detection in lvis_image["detections"]]
    talos_labels = [detection["label"] for detection in talos_image["detections"]]

    print(f"{STR_PREFIX} LVIS labels for image {i+1}: {lvis_labels}")
    print(f"{STR_PREFIX} TALOS labels for image {i+1}: {talos_labels}")

    lvis_label_coincidence_score, lvis_label_coincidences = calculate_lvis_label_coincidence_score(lvis_labels, talos_labels)

    print(f"{STR_PREFIX} LVIS label coincidences in TALOS for image {i+1}: {lvis_label_coincidences}")
    print(f"{STR_PREFIX} LVIS label coincidence score for image {i+1}: {lvis_label_coincidence_score}")

    # Evaluate label coincidence (TALOS labels that are present in LVIS detections)

    talos_label_coincidence_score, talos_label_coincidences = calculate_talos_label_coincidence_score(lvis_labels, talos_labels)

    print(f"{STR_PREFIX} TALOS label coincidences in LVIS for image {i+1}: {talos_label_coincidences}")
    print(f"{STR_PREFIX} TALOS label coincidence score for image {i+1}: {talos_label_coincidence_score}")

    # Evaluate bbox similarity for the detections with label coincidence

    lvis_detections_label_coincidence = []
    talos_detections_label_coincidence = []

    for j, detection in enumerate(lvis_image["detections"]):
        matched = False
        for synonym in detection["labels"]:
            for lvis_label_coincidence in lvis_label_coincidences:
                if synonym in lvis_label_coincidence[0] or lvis_label_coincidence[0] in synonym:
                    lvis_detections_label_coincidence.append(detection)
                    matched = True
                    break
            
            if matched:
                break
    
    for j, detection in enumerate(talos_image["detections"]):
        for talos_label_coincidence in talos_label_coincidences:
            if detection["label"] in talos_label_coincidence[1] or talos_label_coincidence[1] in detection["label"]:
                talos_detections_label_coincidence.append(detection)
                break
    
    bbox_similarity_score, bbox_coincidence_ids = calculate_bbox_similarity_score(
        lvis_detections_label_coincidence,
        talos_detections_label_coincidence
    )

    print(f"{STR_PREFIX} Bbox coincidences for image {i+1}: {bbox_coincidence_ids}")
    print(f"{STR_PREFIX} Bbox similarity score for image {i+1}: {bbox_similarity_score}")
