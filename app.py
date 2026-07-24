from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("model.pt")

@app.get("/")
def home():
    return {"message": "API de détection EPI opérationnelle"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model.predict(image, conf=0.25)
    detections = []

    for box in results[0].boxes:
        detections.append({
            "classe": model.names[int(box.cls[0])],
            "confiance": float(box.conf[0]),
            "boite": box.xyxy[0].tolist()
        })

    return {"detections": detections}
