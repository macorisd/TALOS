# Open Object Classification TLS (Tagging, Location & Segmentation)

Computer Vision pipeline in Python designed for object segmentation in images using Zero-shot open-vocabulary techniques, without relying on predefined categories like those found in datasets such as COCO.

The project is currently under development.

The system follows a structure based on *Tagging, Location & Segmentation*:

## Tagging

Extracts descriptive words for the objects detected in an image.

- **Input:** Image
- **Output:** JSON with detected object instances

Technologies integrated for the Tagging phase in this project:

- **Direct Tagging:**
  - Recognize Anything Plus Model (RAM++)
- **Tagging via LVLM Description and LLM Keyword Extraction:**
  - *Description:*
    - LLaVA LVLM
  - *Keyword Extraction:*
    - DeepSeek LLM

## Location

Identifies the location of bounding boxes for objects present in the image.

- **Inputs:**
  - JSON with detected object instances (output from the Tagging phase)
  - Image (input from the Tagging phase)
- **Outputs:**
  - JSON with bounding box coordinates and detection confidence
  - Input image annotated with bounding boxes and detection confidence

Technologies integrated for the Location phase in this project:

- **Grounding DINO**

## Segmentation

Generates object masks for the detected objects in the image.

- **Inputs:** TODO
- **Outputs:** TODO

Technologies integrated for the Segmentation phase in this project:

- **SAM2**
