# Open Object Classification TLS (Tagging, Location & Segmentation)

Computer Vision pipeline in Python designed for **object detection and segmentation** in images using an **open-vocabulary approach**, without relying on predefined and limited categories like those found in datasets such as COCO.

The project is currently under development. Please check the **develop** branch for the last updates.

The system follows a structure based on three stages: *Tagging, Location & Segmentation*:


## Tagging

Extracts descriptive words (categories) for the objects detected in an image.

- **Input:** Image.
- **Output:** JSON with semantic tags of the detected objects.

Technologies integrated for the Tagging stage in this project:

- **Direct Tagging:**
  - Recognize Anything Plus Model (RAM++)
- **Tagging via LVLM Description and LLM Keyword Extraction:**
  - *LVLM Description:*
    - LLaVA
  - *LLM Keyword Extraction:*
    - DeepSeek


## Location

Identifies the location of bounding boxes for objects present in the image.

- **Inputs:**
  - Image (same input provided to the Tagging stage).
  - JSON with semantic tags of the detected objects (output from the Tagging stage).
- **Outputs:**
  - JSON with bounding box coordinates and detection confidence.
  - Input image annotated with bounding boxes and detection confidence.

Technologies integrated for the Location stage in this project:

- **Grounding DINO**


## Segmentation

Generates instance segmentation masks for the detected objects in the image.

- **Inputs:** 
  - Image (same input provided to the Tagging and Location stage).
  - JSON with bounding box coordinates and detection confidence (output from the Location stage).
- **Output:** Input images with the highlighted segmentation mask for each detected object.

Technologies integrated for the Segmentation stage in this project:

- **SAM2**
