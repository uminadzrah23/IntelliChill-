from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# ✅ Load trained model (IMPORTANT)
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return '''
    <h1>IntelliChill - Upload Image</h1>
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Detect</button>
    </form>
    '''


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"})

        # Save image
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Run YOLO
        results = model(filepath)

        names = model.names
        detected = []
        confidence = 0.0

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    detected.append(names[cls])
                    confidence = float(box.conf[0])
                    break  # ambil first detection sahaja

        print("Detected:", detected)  # DEBUG

        if len(detected) == 0:
            return jsonify({
                "food": None,
                "message": "No object detected"
            })

        food_name = detected[0]

        # ✅ simple status logic (boleh upgrade nanti)
        if confidence > 0.8:
            status = "fresh"
        elif confidence > 0.5:
            status = "check"
        else:
            status = "uncertain"

        return jsonify({
            "food": food_name,
            "confidence": round(confidence, 2),
            "status": status
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)