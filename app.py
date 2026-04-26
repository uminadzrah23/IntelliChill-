import urllib.request
import numpy as np
import cv2
from flask import Flask, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__, static_folder='.', static_url_path='')
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── UPDATE THIS to your phone's IP shown in IP Webcam app ──
CAMERA_URL = 'http://10.26.243.85:8080/stream.mjpg'


def grab_frame_from_stream(url, timeout=12, max_bytes=500_000):
    """
    Robustly grab a single JPEG frame from an MJPEG stream.
    Retries on connection resets and skips corrupt frames.
    """
    req = urllib.request.Request(
        url,
        headers={
            # Some phones reject requests without a browser User-Agent
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Connection': 'keep-alive',
        }
    )

    stream = urllib.request.urlopen(req, timeout=timeout)

    bytes_data = b''
    frames_tried = 0

    while len(bytes_data) < max_bytes:
        try:
            chunk = stream.read(4096)   # larger chunks = fewer read() calls
        except Exception:
            # Stream dropped mid-read — use whatever we buffered
            break

        if not chunk:
            break

        bytes_data += chunk

        # Search for a complete JPEG (starts 0xFF 0xD8, ends 0xFF 0xD9)
        start = bytes_data.find(b'\xff\xd8')
        end   = bytes_data.find(b'\xff\xd9', start + 2 if start != -1 else 0)

        if start != -1 and end != -1 and end > start:
            jpg = bytes_data[start:end + 2]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                stream.close()
                return img                      # ✅ good frame

            # Corrupt frame — discard and keep reading
            bytes_data = bytes_data[end + 2:]
            frames_tried += 1
            if frames_tried > 10:
                break                           # give up after 10 bad frames

    stream.close()
    return None


@app.route("/")
def home():
    return app.send_static_file('index.html')


@app.route('/scan', methods=['GET'])
def scan():
    try:
        img = grab_frame_from_stream(CAMERA_URL)

        if img is None:
            return jsonify({'error': 'Could not grab a valid frame from camera stream. '
                                     'Check that IP Webcam is open and the phone screen is on.'})

        # Run YOLO inference
        results = model(img, conf=0.25)
        names   = model.names
        detected = []

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                detected.append({
                    'name':       names[cls],
                    'confidence': round(conf, 2)
                })

        if not detected:
            return jsonify({'food': None, 'message': 'Nothing detected — try better lighting or move closer.'})

        # Return highest-confidence detection
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        best = detected[0]
        return jsonify({
            'food':       best['name'],
            'confidence': best['confidence'],
            'all':        detected          # bonus: all detections for debugging
        })

    except urllib.error.URLError as e:
        return jsonify({'error': f'Cannot reach camera: {e.reason}. '
                                  'Make sure IP Webcam is running and the IP address is correct.'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
