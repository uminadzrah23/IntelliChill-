import io
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import socketserver
import threading
from http import server
from ultralytics import YOLO

# Load YOUR trained model
model = YOLO('best.pt')

PAGE = """\
<html>
<head><title>IntelliChill - Live Detection</title></head>
<body>
<h1>IntelliChill Food Detection</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame

                    # Convert frame to numpy for YOLO
                    np_frame = np.frombuffer(frame, dtype=np.uint8)
                    img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                    # Run YOLO on every frame
                    results = model(img, conf=0.25)
                    annotated = results[0].plot()  # draws boxes on frame

                    # Convert back to JPEG
                    _, jpeg = cv2.imencode('.jpg', annotated)
                    frame = jpeg.tobytes()

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                print(f"Stream error: {e}")
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8080)
    my_server = StreamingServer(address, StreamingHandler)
    print("Server started at http://10.221.35.85:8080")
    my_server.serve_forever()
finally:
    picam2.stop_recording()
