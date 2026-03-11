import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
from pdf_report import generate_report

# ==================================================
# PATH CONFIG
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

TEMPLATE_FOLDER = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_FOLDER = os.path.join(ROOT_DIR, "frontend", "static")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224
THRESHOLD = 0.5

models = {}
class_maps = {}
positive_class_index = {}

# ==================================================
# LOAD MODELS + AUTO DETECT DISEASE CLASS
# ==================================================

def load_model_and_classes(name):
    model_path = os.path.join(MODEL_FOLDER, f"{name}_model.h5")
    class_path = os.path.join(MODEL_FOLDER, f"{name}_classes.json")

    model = load_model(model_path)

    with open(class_path, "r") as f:
        class_indices = json.load(f)

    reverse_map = {int(v): k for k, v in class_indices.items()}

    # detect disease class (anything not named "normal")
    disease_idx = None
    for idx, cname in reverse_map.items():
        if cname.lower() != "normal":
            disease_idx = idx

    positive_class_index[name] = disease_idx

    print(f"{name} class map:", reverse_map)
    print(f"{name} disease index:", disease_idx)

    return model, reverse_map


for cancer in ["breast", "lung", "skin"]:
    models[cancer], class_maps[cancer] = load_model_and_classes(cancer)

# ==================================================
# PREPROCESS (CORRECT FOR EFFICIENTNET)
# ==================================================

def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)   # 🔥 VERY IMPORTANT
    img = np.expand_dims(img, axis=0)
    return img

# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    cancer_type = request.form["cancer_type"]
    file = request.files["image"]

    if cancer_type not in models:
        return "Invalid cancer type"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = preprocess(filepath)

    model = models[cancer_type]
    class_map = class_maps[cancer_type]
    disease_index = positive_class_index[cancer_type]

    raw_pred = float(model.predict(img)[0][0])
    print("Raw prediction:", raw_pred)

    # ------------------------------------------------
    # Correct prediction logic
    # ------------------------------------------------

    if disease_index == 1:
        predicted_index = 1 if raw_pred >= THRESHOLD else 0
        confidence = raw_pred if predicted_index == 1 else (1 - raw_pred)
    else:
        predicted_index = 0 if raw_pred >= THRESHOLD else 1
        confidence = raw_pred if predicted_index == 0 else (1 - raw_pred)

    confidence = round(confidence * 100, 2)
    result = class_map[predicted_index]

    # ------------------------------------------------
    # Generate PDF
    # ------------------------------------------------

    report_filename = f"report_{filename}.pdf"
    report_path = os.path.join(UPLOAD_FOLDER, report_filename)

    generate_report(report_path, cancer_type, result, confidence, filepath)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        cancer_type=cancer_type,
        image=filename,
        report=report_filename
    )


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)