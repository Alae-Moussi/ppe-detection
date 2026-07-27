import base64
import io
import fastapi
import gradio as gr
from PIL import Image
from ultralytics import YOLO
import spaces  # Import obligatoire pour ZeroGPU

# NOTE : Ne pas charger le modèle au niveau global ici pour ZeroGPU !


def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@spaces.GPU
def process_image(pil_image):
    if pil_image is None:
        return None, [], "⚠️ Aucune image fournie.", "normale"

    # Chargement dynamique sur GPU alloué par ZeroGPU
    model = YOLO("model.pt")

    # Inférence YOLO
    results = model.predict(pil_image, conf=0.12, iou=0.45)

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

    return annotated_pil, detections, statut, criticite


# Application FastAPI
fastapi_app = fastapi.FastAPI(title="PPE Detection API")


@fastapi_app.post("/api/detect")
async def api_detect(file: fastapi.UploadFile = fastapi.File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    annotated_pil, detections, statut, criticite = process_image(image)
    image_base64 = image_to_base64(annotated_pil)

    return {
        "success": True,
        "criticite": criticite,
        "rapport_securite": statut,
        "nb_detections": len(detections),
        "detections": detections,
        "image_annotee": image_base64,
    }


# Interface Gradio
def gradio_wrapper(img):
    annotated_pil, detections, statut, _ = process_image(img)
    return annotated_pil, detections, statut


demo = gr.Interface(
    fn=gradio_wrapper,
    inputs=gr.Image(sources=["webcam", "upload"], type="pil"),
    outputs=[
        gr.Image(type="numpy", label="Détection Visuelle"),
        gr.JSON(label="Détails JSON"),
        gr.Textbox(label="Rapport"),
    ],
)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")