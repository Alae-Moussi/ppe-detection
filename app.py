import gradio as gr
from PIL import Image
import spaces
from ultralytics import YOLO

model = YOLO("model.pt")


@spaces.GPU
def detect(image):
  if image is None:
    return None, [], "⚠️ Aucune image fournie."

  # Inférence avec seuil de confiance plus bas et IOU ajusté
  results = model.predict(image, conf=0.10, iou=0.45)

  annotated_img = results[0].plot()

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

  # Compteurs
  nb_personnes = classes_detectees.count("person")
  nb_no_helmet = classes_detectees.count("no-helmet")
  nb_no_vest = classes_detectees.count("no-vest")
  nb_no_boots = classes_detectees.count("no-boots") + classes_detectees.count(
      "no-shoes"
  )

  avertissements = []

  # Analyse dynamique selon le nombre de détections
  if nb_personnes > 0:
    avertissements.append(f"👥 Personnes détectées : {nb_personnes}")

  if nb_no_helmet > 0:
    avertissements.append(
        f"🚨 DANGER : {nb_no_helmet} personne(s) détectée(s) SANS CASQUE !"
    )

  if nb_no_vest > 0:
    avertissements.append(
        f"🚨 DANGER : {nb_no_vest} personne(s) détectée(s) SANS GILET !"
    )

  if nb_no_boots > 0:
    avertissements.append(
        f"🚨 DANGER : {nb_no_boots} personne(s) détectée(s) SANS BOTTES !"
    )

  # Fallback si des personnes sont là mais qu'aucune classe "helmet" / "no-helmet" n'a été accrochée
  if (
      nb_personnes > 0
      and "helmet" not in classes_detectees
      and nb_no_helmet == 0
  ):
    avertissements.append(
        "⚠️ AVERTISSEMENT : Casque de sécurité non détecté sur l'une des"
        " personnes."
    )

  if not avertissements:
    statut_securite = "✅ SÉCURITÉ CONFORME : Équipements détectés."
  else:
    statut_securite = "\n".join(avertissements)

  return annotated_img, detections, statut_securite


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
        gr.Textbox(label="Rapport de Sécurité", lines=5),
    ],
    title="Système Anti-Accident - Détection d'EPI",
    description="Analyse en temps réel du port des équipements de protection.",
)

if __name__ == "__main__":
  app.launch()
