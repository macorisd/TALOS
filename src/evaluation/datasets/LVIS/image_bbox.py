import os
import cv2

def draw_bounding_boxes(image_name: str, bboxes: list[tuple[int, int, int, int]]):
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "images")
    output_dir = os.path.join(script_dir, "images_bbox")
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, image_name)

    # Check if the image exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Image '{image_name}' not found in '{input_dir}'")

    # Load image
    image = cv2.imread(input_path)

    # Define color and font
    color = (0, 255, 0)  # Green
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1

    # Draw each bounding box
    for i, (xmin, ymin, width, height) in enumerate(bboxes, start=1):
        xmin, ymin = int(xmin), int(ymin)
        xmax, ymax = int(xmin + width), int(ymin + height)
        label = f"bbox{i}"
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(image, label, (xmin, ymin - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)


    # Save output image
    cv2.imwrite(output_path, image)
    print(f"Saved image with bounding boxes to '{output_path}'")

# Example usage
if __name__ == "__main__":
    example_image_name = "76261.jpg"
    example_bboxes = [(
            368.2865905761719,
            180.09201049804688,
            421.1756896972656,
            241.89303588867188
        ), (
            240.44577026367188,
            341.2926025390625,
            302.7044982910156,
            387.7109375
          )]
    draw_bounding_boxes(example_image_name, example_bboxes)
