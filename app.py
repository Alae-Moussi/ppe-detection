import base64
import io
import fastapi
import gradio as gr
from PIL import Image
import spaces
from ultralytics import YOLO

# 1. Chargement du modèle YOLO
model = YOLO("model.pt")


def image_to_base64(pil_img):
  """Convertit une image PIL en chaîne Base64 (JPG) pour l'affichage Frontend."""
  buffered = io.BytesIO()
  pil_img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return f"data:image/jpeg;base64,{img_str}"


@spaces.GPU
def process_image(pil_image):
  if pil_image is None:
    return None, [], "⚠️ Aucune image fournie."

  # Inférence avec seuil abaissé à 0.12 pour capter les gilets/bottes/casques difficiles
  results = model.predict(pil_image, conf=0.12, iou=0.45)

  # Image annotée générée par YOLO (tableau numpy -> PIL)
  annotated_array = results[0].plot()
  annotated_pil = Image.fromarray(annotated_array)

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

  # Décompte des éléments
  nb_personnes = classes_detectees.count("person")
  nb_no_helmet = classes_detectees.count("no-helmet")
  nb_no_vest = classes_detectees.count("no-vest")

  avertissements = []
  criticite = "normale"

  # Analyse d'absence de Casque
  if nb_no_helmet > 0:
    avertissements.append(f"🚨 Absence de casque ({nb_no_helmet} pers.)")
    criticite = "haute"
  elif nb_personnes > 0 and "helmet" not in classes_detectees:
    avertissements.append("🚨 Absence de casque détectée")
    criticite = "haute"

  # Analyse d'absence de Gilet
  if nb_no_vest > 0:
    avertissements.append(f"🚨 Absence de gilet de sécurité ({nb_no_vest} pers.)")
    criticite = "haute"
  elif nb_personnes > 0 and "vest" not in classes_detectees:
    avertissements.append("🚨 Absence de gilet de sécurité")
    criticite = "haute"

  # Synthèse du statut
  if avertissements:
    statut = " | ".join(avertissements)
  elif nb_personnes > 0:
    statut = "✅ Équipements de sécurité conformes."
  else:
    statut = "ℹ️ Aucune personne ou infraction majeure détectée."

  return annotated_pil, detections, statut, criticite


# 2. Application FastAPI pour Laravel / Frontend Vue.js
fastapi_app = fastapi.FastAPI(title="PPE Detection API")


@fastapi_app.post("/api/detect")
async def api_detect(file: fastapi.UploadFile = fastapi.File(...)):
  """Endpoint REST consommé par Laravel / Vue.js."""
  contents = await file.read()
  image = Image.open(io.BytesIO(contents)).convert("RGB")

  annotated_pil, detections, statut, criticite = process_image(image)

  # Convertir l'image dessinée avec les boîtes en Base64
  image_base64 = image_to_base64(annotated_pil)

  return {
      "success": True,
      "criticite": criticite,
      "rapport_securite": statut,
      "nb_detections": len(detections),
      "detections": detections,
      "image_annotee": image_base64,  # Contient l'image avec les cadrages rouges/jaunes
  }


# 3. Interface Gradio (pour démo directe)
def gradio_wrapper(img):
  annotated_pil, detections, statut, _ = process_image(img)
  return annotated_pil, detections, statut


demo = gr.Interface(
    fn=gradio_wrapper,
    inputs=gr.Image(
        sources=["webcam", "upload"],
        type="pil",
        label="Image d'entrée (EPI / Webcam)",
    ),
    outputs=[
        gr.Image(type="numpy", label="Détection Visuelle (Boîtes)"),
        gr.JSON(label="Détails JSON"),
        gr.Textbox(label="Rapport de Sécurité", lines=3),
    ],
    title="Système Anti-Accident - Détection d'EPI",
)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")