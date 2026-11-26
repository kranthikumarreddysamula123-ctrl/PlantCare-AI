
import os
import json
import uuid
import sqlite3
from functools import wraps

import torch
import torch.nn as nn

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    session
)

from torchvision import models, transforms
from PIL import Image
from werkzeug.utils import secure_filename

from gradcam import generate_gradcam
from disease_info import DISEASE_INFO

from history_db import (
    initialize_database,
    save_prediction,
    get_prediction_history,
    register_user,
    login_user,
    get_user_by_id,
    get_all_users
)

from pdf_report import create_prediction_report


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "plantcare-development-key-change-before-deployment"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "prediction_history.db")

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pth"
)

CLASS_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

EVALUATION_PATH = os.path.join(
    BASE_DIR, "models", "evaluation_report.json"
)

TRAINING_HISTORY_PATH = os.path.join(
    BASE_DIR, "models", "training_history.json"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR, "static", "reports"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ==========================================
# 2. AUTHENTICATION DECORATORS
# ==========================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        user = get_user_by_id(session["user_id"])

        if user is None:
            session.clear()
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


def admin_required(function):

    @wraps(function)
    @login_required
    def wrapper(*args, **kwargs):

        if session.get("role") != "admin":
            return "Access denied. Admin privileges required.", 403

        return function(*args, **kwargs)

    return wrapper


# ==========================================
# 3. ADMIN ACCOUNT SETUP
# ==========================================

def create_admin_account():

    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not all([admin_username, admin_email, admin_password]):
        print("Admin setup skipped.")
        print("Set ADMIN_USERNAME, ADMIN_EMAIL and ADMIN_PASSWORD.")
        return

    if len(admin_password) < 12:
        print("Admin setup skipped: password must be at least 12 characters.")
        return

    created = register_user(
        admin_username,
        admin_email.lower(),
        admin_password,
        role="admin"
    )

    if created:
        print("Admin account created successfully!")
    else:
        print("Admin account was not created. Check whether its username or email already exists.")


# ==========================================
# 4. DATABASE INITIALIZATION
# ==========================================

initialize_database()
create_admin_account()

print("PlantCare AI database initialized!")


# ==========================================
# 5. USER REGISTRATION
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            error = "Please fill in all fields."

        elif len(username) < 3:
            error = "Username must contain at least 3 characters."

        elif len(password) < 8:
            error = "Password must contain at least 8 characters."

        elif password != confirm_password:
            error = "Passwords do not match."

        else:

            # Public registration always creates a normal user.
            created = register_user(
                username,
                email,
                password,
                role="user"
            )

            if created:
                return redirect(url_for("login"))

            error = "Username or email already exists."

    return render_template(
        "register.html",
        error=error
    )


# ==========================================
# 6. USER LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = login_user(email, password)

        if user:

            session.clear()

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("home"))

        error = "Invalid email or password."

    return render_template(
        "login.html",
        error=error
    )


# ==========================================
# 7. USER LOGOUT
# ==========================================

@app.route("/logout")
@login_required
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==========================================
# 8. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 9. LOAD CLASS NAMES
# ==========================================

with open(CLASS_PATH, "r") as file:
    classes = json.load(file)

print("Total classes loaded:", len(classes))


# ==========================================
# 10. LOAD RESNET18 MODEL
# ==========================================

model = models.resnet18(weights=None)

input_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(input_features, len(classes))
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Plant disease detection model loaded successfully!")


# ==========================================
# 11. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 12. CHECK FILE EXTENSION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 13. PREDICT IMAGE AND GENERATE GRAD-CAM
# ==========================================

def predict_image(image_path):

    original_image = Image.open(image_path).convert("RGB")

    input_tensor = transform(original_image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    predicted_index, confidence, heatmap, overlay = generate_gradcam(
        model,
        input_tensor,
        original_image,
        device
    )

    disease = classes[predicted_index]

    confidence = round(float(confidence), 2)

    unique_id = uuid.uuid4().hex

    original_filename = os.path.basename(image_path)

    heatmap_filename = (
        f"gradcam_{unique_id}_{original_filename}"
    )

    overlay_filename = (
        f"overlay_{unique_id}_{original_filename}"
    )

    heatmap_save_path = os.path.join(
        UPLOAD_FOLDER,
        heatmap_filename
    )

    overlay_save_path = os.path.join(
        UPLOAD_FOLDER,
        overlay_filename
    )

    Image.fromarray(heatmap).save(heatmap_save_path)
    Image.fromarray(overlay).save(overlay_save_path)

    return (
        disease,
        confidence,
        heatmap_filename,
        overlay_filename
    )


# ==========================================
# 14. HOME PAGE AND IMAGE PREDICTION
# ==========================================

@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    result = None
    confidence = None

    image_path = None
    heatmap_path = None
    overlay_path = None

    disease_details = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:

            error = "Please select an image."

        else:

            file = request.files["file"]

            if file.filename == "":

                error = "No image selected."

            elif not allowed_file(file.filename):

                error = "Only JPG, JPEG and PNG images are allowed."

            else:

                try:

                    unique_filename = (
                        uuid.uuid4().hex
                        + "_"
                        + secure_filename(file.filename)
                    )

                    save_path = os.path.join(
                        UPLOAD_FOLDER,
                        unique_filename
                    )

                    file.save(save_path)

                    (
                        result,
                        confidence,
                        heatmap_filename,
                        overlay_filename
                    ) = predict_image(save_path)

                    image_path = "uploads/" + unique_filename

                    heatmap_path = "uploads/" + heatmap_filename

                    overlay_path = "uploads/" + overlay_filename

                    disease_details = DISEASE_INFO.get(
                        result,
                        {
                            "title": result.replace("_", " "),
                            "status": "Information unavailable",
                            "description": "No additional information available.",
                            "symptoms": [],
                            "causes": [],
                            "prevention": [],
                            "management": []
                        }
                    )

                    save_prediction(
                        image_path,
                        result,
                        confidence,
                        heatmap_path,
                        overlay_path,
                        user_id=session["user_id"]
                    )

                    print("Prediction saved for user:", session["username"])

                except Exception as e:

                    print("Prediction error:", str(e))

                    error = (
                        "Unable to process the image. "
                        "Please upload another clear plant leaf image."
                    )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
        disease_details=disease_details,
        error=error
    )


# ==========================================
# 15. PREDICTION HISTORY
# ==========================================

@app.route("/history")
@login_required
def history():

    is_admin = session.get("role") == "admin"

    records = get_prediction_history(
        user_id=session["user_id"],
        is_admin=is_admin
    )

    return render_template(
        "history.html",
        records=records
    )


# ==========================================
# 16. MODEL PERFORMANCE DASHBOARD
# ==========================================

@app.route("/performance", methods=["GET"])
@login_required
def performance():

    try:

        if not os.path.exists(EVALUATION_PATH):
            return (
                "Evaluation report not found. "
                "Please run python evaluate.py first.",
                404
            )

        if not os.path.exists(TRAINING_HISTORY_PATH):
            return (
                "Training history not found in the models folder.",
                404
            )

        with open(EVALUATION_PATH, "r") as file:
            evaluation = json.load(file)

        with open(TRAINING_HISTORY_PATH, "r") as file:
            training_history = json.load(file)

        return render_template(
            "performance.html",
            metrics=evaluation["metrics"],
            test_images=evaluation["test_images"],
            class_names=evaluation["class_names"],
            class_report=evaluation["class_wise_report"],
            confusion_matrix=evaluation["confusion_matrix"],
            training_history=training_history
        )

    except Exception as e:

        print("Performance dashboard error:", str(e))

        return (
            "Unable to load Performance Dashboard. "
            "Check the Flask terminal for details.",
            500
        )


# ==========================================
# 17. ADMIN DASHBOARD
# ==========================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    users = get_all_users()

    records = get_prediction_history(is_admin=True)

    return render_template(
        "admin.html",
        users=users,
        records=records
    )


# ==========================================
# 18. DOWNLOAD PREDICTION REPORT
# ==========================================

@app.route("/download-report", methods=["POST"])
@login_required
def download_report():

    try:

        disease = request.form.get("disease")
        confidence = request.form.get("confidence")

        image_path = request.form.get("image_path")
        heatmap_path = request.form.get("heatmap_path")
        overlay_path = request.form.get("overlay_path")

        if not all([
            disease,
            confidence,
            image_path,
            heatmap_path,
            overlay_path
        ]):
            return "Missing prediction information.", 400

        # Confirm that this prediction belongs to the logged-in user.
        # Admins can download reports for any stored prediction.

        connection = sqlite3.connect(DATABASE)

        if session.get("role") == "admin":

            record = connection.execute("""
                SELECT id
                FROM predictions
                WHERE image_path = ?
                  AND heatmap_path = ?
                  AND overlay_path = ?
                  AND disease = ?
                  AND confidence = ?
                LIMIT 1
            """, (
                image_path,
                heatmap_path,
                overlay_path,
                disease,
                confidence
            )).fetchone()

        else:

            record = connection.execute("""
                SELECT id
                FROM predictions
                WHERE user_id = ?
                  AND image_path = ?
                  AND heatmap_path = ?
                  AND overlay_path = ?
                  AND disease = ?
                  AND confidence = ?
                LIMIT 1
            """, (
                session["user_id"],
                image_path,
                heatmap_path,
                overlay_path,
                disease,
                confidence
            )).fetchone()

        connection.close()

        if record is None:
            return "Prediction not found or access denied.", 403

        # Only allow image files inside static/uploads.

        upload_root = os.path.realpath(UPLOAD_FOLDER)

        def resolve_upload_path(relative_path):

            if not relative_path.startswith("uploads/"):
                raise ValueError("Invalid image path.")

            full_path = os.path.realpath(
                os.path.join(
                    BASE_DIR,
                    "static",
                    relative_path
                )
            )

            if os.path.commonpath([
                upload_root,
                full_path
            ]) != upload_root:

                raise ValueError("Invalid image path.")

            if not os.path.isfile(full_path):
                raise FileNotFoundError(
                    "Prediction image not found."
                )

            return full_path

        original_image = resolve_upload_path(image_path)
        heatmap_image = resolve_upload_path(heatmap_path)
        overlay_image = resolve_upload_path(overlay_path)

        disease_details = DISEASE_INFO.get(
            disease,
            {
                "title": disease.replace("_", " "),
                "description": "Information not available.",
                "symptoms": [],
                "causes": [],
                "prevention": [],
                "management": []
            }
        )

        filename = (
            f"PlantCare_Report_{uuid.uuid4().hex[:8]}.pdf"
        )

        report_path = os.path.join(
            REPORT_FOLDER,
            filename
        )

        create_prediction_report(
            output_path=report_path,
            disease=disease,
            confidence=confidence,
            original_image=original_image,
            heatmap_image=heatmap_image,
            overlay_image=overlay_image,
            disease_details=disease_details
        )

        return send_file(
            report_path,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:

        print("PDF generation error:", str(e))

        return (
            "Unable to generate PDF report. "
            "Check the VS Code terminal for details.",
            500
        )


# ==========================================
# 19. FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. Maximum allowed size is 10 MB.",
        413
    )


# ==========================================
# 20. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("PLANTCARE AI SERVER")
    print("================================")

    print("Login: http://127.0.0.1:5000/login")
    print("Register: http://127.0.0.1:5000/register")
    print("Home: http://127.0.0.1:5000/")
    print("History: http://127.0.0.1:5000/history")
    print("Performance: http://127.0.0.1:5000/performance")
    print("Admin: http://127.0.0.1:5000/admin")

    print("================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
