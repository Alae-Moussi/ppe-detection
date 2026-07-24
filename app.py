import gradio as gr
from PIL import Image
import spaces
from ultralytics import YOLO

# Chargement du modèle
model = YOLO("model.pt")


@spaces.GPU
def detect(image):
  if image is None:
    return None, [], "⚠️ Aucune image fournie."

  # Inférence YOLO
  results = model.predict(image, conf=0.25)

  # 1. Image annotée avec les boîtes
  annotated_img = results[0].plot()

  # 2. Extraction du JSON + Analyse des avertissements (Warning)
  detections = []
  classes_detectees = []

  for box in results[0].boxes:
    nom_classe = model.names[int(box.cls[0])]
    classes_detectees.append(nom_classe.lower())

    detections.append({
        "classe": nom_classe,
        "confiance": round(float(box.conf[0]), 3),
        "boite": [round(coord, 1) for coord in box.xyxy[0].tolist()],
    })

  # 3. Logique de détection des risques (Avertissements)
  avertissements = []

  # Vérification du casque (helmet)
  if "no-helmet" in classes_detectees:
    avertissements.append("🚨 DANGER : Personne détectée SANS CASQUE !")
  elif "person" in classes_detectees and "helmet" not in classes_detectees:
    avertissements.append(
        "⚠️ AVERTISSEMENT : Personne détectée mais pas de casque visible."
    )

  # Vérification du gilet (vest)
  if "no-vest" in classes_detectees:
    avertissements.append("🚨 DANGER : Personne détectée SANS GILET !")
  elif "person" in classes_detectees and "vest" not in classes_detectees:
    avertissements.append(
        "⚠️ AVERTISSEMENT : Personne détectée mais pas de gilet visible."
    )

  # Résumé de l'état de sécurité
  if not avertissements:
    if "person" in classes_detectees:
      statut_securite = (
          "✅ SÉCURITÉ CONFORME : Tous les EPI requis sont portés !"
      )
    else:
      statut_securite = "ℹ️ Aucune personne détectée sur l'image."
  else:
    statut_securite = "\n".join(avertissements)

  return annotated_img, detections, statut_securite


# Interface Gradio
app = gr.Interface(
    fn=detect,
    inputs=gr.Image(
        sources=["webcam", "upload"],
        type="pil",
        label="Image d'entrée (EPI / Webcam)",
    ),
    outputs=[
        gr.Image(type="numpy", label="Détection Visuelle"),
        gr.JSON(label="Détails JSON (Pour API / Laravel)"),
        gr.Textbox(label="Alerte & Sécurité (Warning)", lines=3),
    ],
    title="Système Anti-Accident - Détection d'EPI",
    description="Analyse automatique de la conformité du port des équipements de protection individuelle en temps réel.",
)

if __name__ == "__main__":
  app.launch()
