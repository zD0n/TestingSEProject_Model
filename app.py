from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io, base64, cv2
import numpy as np

app = Flask(__name__)
CORS(app)

model = YOLO("yolo11n.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model.predict(image, conf=0.8)

    # Bounding box image
    im = results[0].plot()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(im)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")

    img_base64 = base64.b64encode(buf.getvalue()).decode()

    detections = []
    if results[0].boxes is not None:
        for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
            detections.append({
                "class": results[0].names[int(cls_id)],
                "confidence": float(conf)
            })

    return jsonify({
        "detections": detections,
        "image": img_base64
    })