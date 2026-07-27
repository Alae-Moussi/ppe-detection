import base64
import io
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# 1. Chargement du modèle
model = YOLO("model.pt")


def process_image(pil_image):
  if pil_image is None:
    return None, [], "⚠️ Aucune image fournie.", "normale", ""

  # Inférence YOLO
  results = model.predict(pil_image, conf=0.12, iou=0.45)

  annotated_array = results[0].plot()
  annotated_pil = Image.fromarray(annotated_array)

  # Conversion en Base64
  buffered = io.BytesIO()
  annotated_pil.save(buffered, format="JPEG")
  b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  image_base64 = f"data:image/jpeg;base64,{b64_str}"

  detections = []
  classes_detectees = []

  for box in results[0].boxes:
    nom_classe = model.names[int(box.cls[0])].lower()
    classes_detectees.append(nom_classe)

    detections.append({
        "classe": model.names[int(box.cls[0])],
        "confiance": round(float(box.conf[0]), 3),
        "boite": [round(coord, 1) for coord in box.xyxy[0].tolist()],
    })

  nb_personnes = classes_detectees.count("person")
  nb_no_helmet = classes_detectees.count("no-helmet")
  nb_no_vest = classes_detectees.count("no-vest")

  avertissements = []
  criticite = "normale"

  if nb_no_helmet > 0 or (
      nb_personnes > 0 and "helmet" not in classes_detectees
  ):
    avertissements.append("🚨 Absence de casque")
    criticite = "haute"

  if nb_no_vest > 0 or (nb_personnes > 0 and "vest" not in classes_detectees):
    avertissements.append("🚨 Absence de gilet de sécurité")
    criticite = "haute"

  if avertissements:
    statut = " | ".join(avertissements)
  elif nb_personnes > 0:
    statut = "✅ Équipements conformes."
  else:
    statut = "ℹ️ Aucune personne détectée."

  return annotated_pil, detections, statut, criticite, image_base64


# Interface Gradio standard (reconnue nativement par Hugging Face)
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(sources=["webcam", "upload"], type="pil"),
    outputs=[
        gr.Image(type="numpy", label="Détection Visuelle"),
        gr.JSON(label="Détails JSON"),
        gr.Textbox(label="Rapport"),
        gr.Textbox(label="Criticite"),
        gr.Textbox(label="Image Base64"),
    ],
    title="PPE Detection API",
)

demo.launch()