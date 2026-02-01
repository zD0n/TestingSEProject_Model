from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io, base64, cv2

app = Flask(__name__)
CORS(app)

# Load model once
model = YOLO("yolo11n.pt")
model.to("cpu")

@app.route("/", methods=["GET"])
def health():
    return "YOLO API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    image.thumbnail((640, 640))

    results = model.predict(
        image,
        conf=0.8,
        device="cpu",
        verbose=False
    )

    im = results[0].plot()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format="JPEG")

    detections = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for c, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
            detections.append({
                "class": results[0].names[int(c)],
                "confidence": float(conf)
            })

    return jsonify({
        "detections": detections,
        "image": base64.b64encode(buf.getvalue()).decode()
    })
