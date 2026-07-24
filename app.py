import gradio as gr
from PIL import Image
import spaces
from ultralytics import YOLO

model = YOLO("model.pt")


# On ajoute le décorateur ici pour exploiter ZeroGPU
@spaces.GPU
def detect(image):
  if image is None:
    return None, []

  results = model.predict(image, conf=0.25)
  annotated_img = results[0].plot()

  detections = []
  for box in results[0].boxes:
    detections.append({
        "classe": model.names[int(box.cls[0])],
        "confiance": round(float(box.conf[0]), 3),
        "boite": [round(coord, 1) for coord in box.xyxy[0].tolist()],
    })

  return annotated_img, detections


app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Image d'entrée (EPI)"),
    outputs=[
        gr.Image(type="numpy", label="Détection Visuelle"),
        gr.JSON(label="Résultats JSON (API)"),
    ],
    title="Détection d'EPI - YOLOv8s-World",
)

if __name__ == "__main__":
  app.launch()
