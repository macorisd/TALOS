import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load SAM2 predictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# Get input image path
script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir, "..", "input_images", "desk.jpg")

# Load and convert the image to RGB
image_bgr = cv2.imread(input_image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Define bounding boxes
bboxes = [
    [198.3580, 280.5756, 845.7460, 773.1286],
    [563.6533, 3.0841, 1290.2542, 820.1292],
    [808.2595, 629.6970, 936.9216, 771.4504],
    [769.8670, 606.9142, 889.4572, 731.3263],
    [912.3726, 667.9551, 1010.9000, 808.9399],
    [770.0927, 606.8397, 1086.3584, 830.1667],
    [347.5200, 346.8821, 576.4768, 458.1528],
    [3.3774, 1.9231, 423.2549, 531.5089],
    [196.3197, 284.0512, 847.8402, 774.0870],
    [370.4004, 489.9409, 480.4674, 596.0664],
    [971.8371, 695.0385, 1087.9807, 829.1167],
    [498.7555, 224.4081, 569.6581, 342.9078],
    [382.5399, 320.5384, 502.0479, 378.7103],
    [261.2338, 348.0685, 338.9390, 389.9977],
    [330.9445, 460.1396, 419.1620, 527.0400],
    [2.4708, 2.1932, 579.5705, 532.0803],
    [540.8918, 335.6943, 645.5584, 384.0859]
]

# Iterate over each bounding box
for i, bbox in enumerate(bboxes):
    # Perform segmentation with SAM2
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=[bbox])

    # Create an overlay image with randomly colored masks
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    for mask in masks:
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)  # Random color
        mask_bool = mask.astype(bool)  # Convert to boolean
        mask_overlay[mask_bool] = color

    # Merge the original image with the segmentation
    alpha = 0.5  # Transparency level
    overlayed_image = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)

    # Save the image in script_dir
    output_image_path = os.path.join(script_dir, f"segmented_desk_bbox_{i}.jpg")
    cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image for bbox {i} saved at: {output_image_path}")

    # Display the segmented image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlayed_image)
    plt.axis("off")
    plt.show()