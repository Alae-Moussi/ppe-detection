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

  # Inférence YOLO avec seuil ajusté (conf=0.15 pour attraper plus de détections)
  results = model.predict(image, conf=0.15)

  # 1. Image annotée
  annotated_img = results[0].plot()

  # 2. Construction du JSON de sortie
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

  # 3. Logique d'alerte pour CASQUE, GILET et BOTTES
  avertissements = []

  # CASQUE (Helmet)
  if "no-helmet" in classes_detectees:
    avertissements.append("🚨 DANGER : Personne détectée SANS CASQUE !")
  elif "person" in classes_detectees and "helmet" not in classes_detectees:
    avertissements.append(
        "⚠️ AVERTISSEMENT : Personne présente sans casque visible."
    )

  # GILET (Vest)
  if "no-vest" in classes_detectees:
    avertissements.append("🚨 DANGER : Personne détectée SANS GILET !")
  elif "person" in classes_detectees and "vest" not in classes_detectees:
    avertissements.append(
        "⚠️ AVERTISSEMENT : Personne présente sans gilet visible."
    )

  # BOTTES (Boots)
  if "no-boots" in classes_detectees or "no-shoes" in classes_detectees:
    avertissements.append(
        "🚨 DANGER : Personne détectée SANS BOTTES DE SÉCURITÉ !"
    )
  elif "person" in classes_detectees and (
      "boots" not in classes_detectees and "shoes" not in classes_detectees
  ):
    avertissements.append(
        "⚠️ AVERTISSEMENT : Bottes de sécurité non détectées."
    )

  # Synthèse du statut
  if avertissements:
    statut_securite = "\n".join(avertissements)
  elif (
      "helmet" in classes_detectees
      or "vest" in classes_detectees
      or "boots" in classes_detectees
  ):
    statut_securite = (
        "✅ SÉCURITÉ CONFORME : Équipements de protection détectés !"
    )
  else:
    statut_securite = "ℹ️ Analyse terminée : Aucun équipement ou infraction majeure détecté."

  return annotated_img, detections, statut_securite


# Interface Gradio avec Webcam et Upload
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
        gr.Textbox(label="Alerte & Sécurité (Warning)", lines=4),
    ],
    title="Système Anti-Accident - Détection d'EPI",
    description="Analyse en temps réel du port des équipements de protection (Casque, Gilet, Bottes).",
)

if __name__ == "__main__":
  app.launch()
