import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

# Configuración
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar procesador y modelo
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Ruta de la imagen
script_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_directory, "2.jpg")
image = Image.open(image_path)

# Texto de consulta
# text = "shoes. desk. sunglasses. paper sheets. curtain."
text = "slipper. curtain. table. electronic. floor. office supply. room. sandal. shoe. stool"


# Procesamiento y predicción
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-procesamiento
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    target_sizes=[image.size[::-1]]
)[0]

# Dibujar los bounding boxes en la imagen
draw = ImageDraw.Draw(image)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 20)

for score, box, label in zip(results["scores"], results["boxes"], results["text_labels"]):
    if score > 0.3:  # Filtrar por confianza
        box = [int(b) for b in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red", font=font)

# Guardar la imagen resultante
output_path = os.path.join(script_directory, "output.jpg")
image.save(output_path)

print(f"Imagen guardada en: {output_path}")
