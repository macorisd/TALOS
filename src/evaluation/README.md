# TALOS Evaluation Metrics

## Metrics Overview

A quantitative evaluation has been carried out to compare the different model combinations integrated into TALOS. This evaluation is based on five key metrics that assess the system's performance on detection and segmentation results, yielding an average final score. Additionally, the average execution time is reported for each tested pipeline configuration.

Specifically, six possible TALOS model combinations were evaluated and their results compared against the ground truth of the LVIS dataset, which provides detailed object instance annotations and segmentation masks. The LVIS data was carefully adapted to match the TALOS output format for direct comparison. A random subset of 1,000 images was processed by TALOS, and the semantic instance segmentation results were assessed using the following five metrics:

* **Detection count**: Measures the difference between the number of object detections produced by TALOS and the number of instances in the ground truth. The score is computed using a custom function, penalizing large discrepancies between detected and actual object counts.
* **Label precision**: Quantifies the proportion of semantic labels detected by TALOS that are also present in the ground truth, reflecting the accuracy in the assignment of object categories.
* **Label recall**: Measures the proportion of ground truth semantic labels that are correctly identified by TALOS, indicating the system’s ability to recover all relevant categories present in the image.
* **Bbox similarity**: Assesses the localization quality of detected bounding boxes for matched labels, using the Intersection over Union (IoU) metric between predicted and ground truth boxes.
* **Mask similarity**: Evaluates the quality of the binary segmentation masks for matched bounding boxes, again using the IoU metric to compare predicted and ground truth masks.

The average final score is computed as the mean of the five previous metrics. Additionally, the average inference time per pipeline configuration is reported.

---

## Results

Please note that only the Tagging models are shown in the evaluation results, as the Location stage for every case was performed using Grounding DINO, and the Segmentation stage was performed using SAM2 (Segment Anything Model 2).

| TAGGING MODEL(S)             | Detection count | Label precision | Label recall | BBox sim. | Mask sim. | Avg final score | Avg exec. time (s) |
|------------------------------|----------------|----------------|--------------|-----------|-----------|-----------------|--------------------|
| MiniCPM                      | 68.88          | 40.88          | 41.92        | 75.26     | 71.09     | 59.61           | 1.63               |
| Gemma                        | 58.80          | 30.79          | 58.73        | 75.43     | 71.99     | 59.15           | 8.11               |
| Qwen                         | 58.01          | 29.78          | 59.73        | 74.90     | 72.02     | 58.89           | 4.18               |
| MiniCPM + MiniCPM            | 65.34          | 27.65          | 47.08        | 74.68     | 72.28     | 57.41           | 2.68               |
| LLaVA + MiniCPM              | 64.92          | 28.60          | 43.58        | 74.78     | 70.88     | 56.55           | 5.63               |
| RAM Plus                     | 66.22          | 26.32          | 49.50        | 71.15     | 68.86     | 56.41           | 0.63               |
