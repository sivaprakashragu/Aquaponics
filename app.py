from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import serial
import threading
import time
import json

# ── ML ──
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# ───────────────────────────────
# LOAD ML MODEL
# ───────────────────────────────
model = load_model("model.keras")

classes = [
    "Aeromoniasis", "Gill_Disease", "Healthy",
    "Parasitic", "Red_Disease", "Saprolegniasis", "White_Tail"
]

solutions = {
    "Aeromoniasis":   "Improve water quality and use antibiotics.",
    "Gill_Disease":   "Increase aeration and reduce ammonia.",
    "Healthy":        "Fish is healthy. Maintain tank conditions.",
    "Parasitic":      "Use anti-parasitic treatment.",
    "Red_Disease":    "Start antibacterial treatment immediately.",
    "Saprolegniasis": "Apply antifungal treatment.",
    "White_Tail":     "Improve hygiene and isolate fish."
}

# ───────────────────────────────
# SENSOR CONFIG
# ───────────────────────────────
PORT = 'COM3'   # ⚠️ CHANGE if needed
BAUD = 9600

STALE_AFTER = 8

# Shared state
lock = threading.Lock()

latest_temp = None
latest_depth = None

last_temp_time = None
last_depth_time = None

temp_status = "connecting"
depth_status = "connecting"


# ───────────────────────────────
# SERIAL THREAD (FIXED)
# ───────────────────────────────
def read_serial():
    global latest_temp, latest_depth
    global last_temp_time, last_depth_time
    global temp_status, depth_status

    ser = None

    while True:
        try:
            if ser is None or not ser.is_open:
                print("[serial] Connecting...")
                ser = serial.Serial(PORT, BAUD, timeout=1)
                time.sleep(2)
                print("[serial] Connected ✓")

            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            # 🔥 DEBUG (important)
            print("[RAW]", line)

            # ── TEMPERATURE ──
            if line.startswith("Temperature:"):
                try:
                    value = line.replace("Temperature:", "").strip()
                    temp = float(value)

                    with lock:
                        latest_temp = temp
                        last_temp_time = time.time()
                        temp_status = "ok"

                    print(f"[serial] 🌡 {temp} °C")

                except:
                    print("[ERROR] Temp parse failed:", line)

            # ── DISTANCE ──
            elif line.startswith("Distance:"):
                try:
                    value = line.replace("Distance:", "").strip()
                    dist = float(value)

                    with lock:
                        latest_depth = dist
                        last_depth_time = time.time()
                        depth_status = "ok"

                    print(f"[serial] 📡 {dist} cm")

                except:
                    print("[ERROR] Distance parse failed:", line)

        except Exception as e:
            print("[serial ERROR]", e)

            temp_status = "error"
            depth_status = "error"

            try:
                ser.close()
            except:
                pass

            ser = None
            time.sleep(2)


# Start thread
threading.Thread(target=read_serial, daemon=True).start()


# ───────────────────────────────
# ML PREPROCESS
# ───────────────────────────────
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    return img


# ───────────────────────────────
# ROUTES
# ───────────────────────────────
@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/disease')
def disease():
    return render_template("disease.html")

@app.route('/aquaponics')
def aquaponics():
    return render_template("aquaponics.html")

@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route('/dimension')
def dimension():
    return render_template("dimension.html")

@app.route('/list')
def list_page():
    return render_template("list.html")


# ───────────────────────────────
# API: TEMPERATURE
# ───────────────────────────────
@app.route('/api/temperature')
def get_temperature():
    with lock:
        temp = latest_temp
        last = last_temp_time
        status = temp_status

    age = round(time.time() - last, 1) if last else None

    if age and age > STALE_AFTER:
        status = "stale"

    return jsonify({
        "temperature": temp,
        "status": status,
        "age_seconds": age
    })


# ───────────────────────────────
# API: DEPTH
# ───────────────────────────────
@app.route('/api/depth')
def get_depth():
    with lock:
        depth = latest_depth
        last = last_depth_time
        status = depth_status

    age = round(time.time() - last, 1) if last else None

    if age and age > STALE_AFTER:
        status = "stale"

    return jsonify({
        "distance_cm": depth,
        "status": status,
        "age_seconds": age
    })


# ───────────────────────────────
# ML PREDICTION
# ───────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        img = preprocess(img)

        prediction = model.predict(img)[0]
        idx = int(np.argmax(prediction))

        return jsonify({
            "disease": classes[idx],
            "confidence": f"{prediction[idx]*100:.2f}",
            "solution": solutions[classes[idx]]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ───────────────────────────────
# RUN
# ───────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)