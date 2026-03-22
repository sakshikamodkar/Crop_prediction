import datetime
import os
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# Safe Model Loader
# -------------------------
def safe_load_model(file_path, default=None):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"⚠️ Could not load {file_path}: {e}")
        return default

watering_model = safe_load_model(os.path.join(BASE_DIR, "watering_model.pkl"))
fertilizer_model = safe_load_model(os.path.join(BASE_DIR, "fertilizer_model.pkl"))
le_watering = safe_load_model(os.path.join(BASE_DIR, "le_watering.pkl"))
le_fertilizer = safe_load_model(os.path.join(BASE_DIR, "le_fertilizer.pkl"))

try:
    cnn_model = load_model(os.path.join(BASE_DIR, "crop_image_model.h5"))
    with open(os.path.join(BASE_DIR, "image_labels.txt"), "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print("⚠️ CNN Model or labels missing:", e)
    cnn_model = None
    class_labels = []

# -------------------------
# Global Data Log
# -------------------------
data_log = []

# -------------------------
# Helper Functions
# -------------------------
def classify_soil_condition(soil_moisture):
    try:
        m = float(soil_moisture)
    except Exception:
        return "Unknown"
    if m < 20:
        return "Dry"
    elif m <= 60:
        return "Optimal"
    return "Wet"

def sensor_based_pest_risk(temp, hum, soil, rain, light):
    score = 0
    if hum >= 75:
        score += 2
    if 20 <= temp <= 30:
        score += 1
    if 30 <= soil <= 60:
        score += 1
    if rain == 1:
        score += 1

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"

def combined_pest_risk(cnn_label, conf, sensor_entry):
    disease_keywords = ["blight", "scab", "rust", "spot", "virus", "mite", "aphid"]
    is_disease = any(k in cnn_label.lower() for k in disease_keywords)
    sensor_risk = sensor_entry.get("pest_risk_sensor", "Low") if sensor_entry else "Low"

    if is_disease and conf >= 60:
        return "High"
    if is_disease and conf >= 30:
        return "Medium"
    if sensor_risk == "High":
        return "High"
    if sensor_risk == "Medium":
        return "Medium"
    return "Low"

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    """Main Dashboard"""
    latest = data_log[-1] if data_log else {}
    return render_template(
        "dashboard.html",
        records=list(reversed(data_log[-10:])),
        current=latest,
        sensor_data=data_log
    )

@app.route("/data", methods=["POST"])
def receive_data():
    """Receive IoT Sensor Data"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Body must be JSON"}), 400

    try:
        data = request.get_json(force=True)

        # Timestamp
        data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Normalize
        data["temperature"] = float(data.get("temperature", 0))
        data["humidity"] = float(data.get("humidity", 0))
        data["moisture"] = float(data.get("moisture", 0))
        data["light"] = float(data.get("light", 0))
        data["rain"] = int(data.get("rain", 0))
        data["ph"] = float(data.get("ph", 7.0))
        data["crop"] = str(data.get("crop", "Wheat"))

        # Predictions
        if watering_model and fertilizer_model:
            df = pd.DataFrame([{
                "Temperature": data["temperature"],
                "Humidity": data["humidity"],
                "SoilMoisture": data["moisture"],
                "Rain": data["rain"],
                "Light": data["light"],
                "Crop": data["crop"]
            }])

            w_pred = watering_model.predict(df)[0]
            f_pred = fertilizer_model.predict(df)[0]

            data["watering_prediction"] = (
                le_watering.inverse_transform([w_pred])[0] if le_watering else str(w_pred)
            )
            data["fertilizer_prediction"] = (
                le_fertilizer.inverse_transform([f_pred])[0] if le_fertilizer else str(f_pred)
            )
        else:
            data["watering_prediction"] = "Model Missing"
            data["fertilizer_prediction"] = "Model Missing"

        # Sensor logic
        data["soil_condition"] = classify_soil_condition(data["moisture"])
        data["pest_risk_sensor"] = sensor_based_pest_risk(
            data["temperature"],
            data["humidity"],
            data["moisture"],
            data["rain"],
            data["light"]
        )

        data_log.append(data)
        if len(data_log) > 100:
            data_log.pop(0)

        print(f"✅ Sensor Data Received: {data}")
        return jsonify({"status": "ok", "data": data}), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle Crop Image Upload + CNN Classification"""
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(file_path)

    if cnn_model is None or not class_labels:
        print("⚠️ CNN model not loaded; skipping prediction.")
        return redirect(url_for("index"))

    # Predict
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    preds = cnn_model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = round(float(np.max(preds[0]) * 100), 2)
    crop_name = class_labels[class_idx]

    latest_sensor = data_log[-1] if data_log else {}
    pest_risk = combined_pest_risk(crop_name, confidence, latest_sensor)

    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cnn_prediction": crop_name,
        "confidence": confidence,
        "image_path": file_path,
        "pest_risk_combined": pest_risk
    }

    data_log.append(record)
    if len(data_log) > 100:
        data_log.pop(0)

    print(f"🖼️ Image Classified: {record}")
    return redirect(url_for("index"))

@app.route("/latest")
def latest_data():
    """Get Latest Data in JSON"""
    if not data_log:
        return jsonify({"status": "empty"}), 200
    return jsonify({"status": "ok", "data": data_log[-1]}), 200

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
