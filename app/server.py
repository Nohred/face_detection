import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import joblib
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Reduce TF log noise (must be set before importing TF in some setups)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.functions import detect_faces, find_camera_index, get_color, get_embedding

INDEX_HTML_PATH = APP_DIR / "index.html"
STYLE_CSS_PATH = APP_DIR / "style.css"

FACENET_MODEL_PATH = PROJECT_DIR / "model" / "facenet_keras.h5"
CLASSIFIER_PATH = PROJECT_DIR / "model" / "face_classifier.joblib"
LABEL_ENCODER_PATH = PROJECT_DIR / "model" / "label_encoder.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clamp_float(value, lo, hi, default):
    try:
        v = float(value)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _clamp_int(value, lo, hi, default):
    try:
        v = int(value)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _safe_str(value, default=""):
    if value is None:
        return default
    return str(value)


def sanitize_update(payload, current):
    updated = dict(current)

    if "face_detector" in payload:
        backend = _safe_str(payload.get("face_detector"), updated["face_detector"]).strip().lower()
        if backend in ("haar", "mtcnn"):
            updated["face_detector"] = backend

    if "process_every_n_frames" in payload:
        updated["process_every_n_frames"] = _clamp_int(
            payload.get("process_every_n_frames"), 1, 60, updated["process_every_n_frames"]
        )

    if "max_faces" in payload:
        updated["max_faces"] = _clamp_int(payload.get("max_faces"), 0, 20, updated["max_faces"])

    if "draw_color" in payload:
        updated["draw_color"] = _safe_str(payload.get("draw_color"), updated["draw_color"])

    if "draw_thickness" in payload:
        updated["draw_thickness"] = _clamp_int(payload.get("draw_thickness"), 1, 10, updated["draw_thickness"])

    return updated


def make_info_frame(text, width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


class CameraWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()

        self._config = {
            "camera_index": None,
            "frame_width": 640,
            "frame_height": 480,
            "capture_fps": 30,
            "face_detector": "haar",
            "max_faces": 5,
            "process_every_n_frames": 1,
            "draw_color": "#00FF00",
            "draw_thickness": 2,
        }

        self._latest_jpeg = None
        self._frame_id = 0
        self._browser_frame = None
        self._browser_frame_id = 0

        self._cap = None
        self._cap_settings = None
        self._capture_error = None

        self._frame_index = 0
        self._cached_predictions = []  # (x1, y1, x2, y2, label, conf)

        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.time()

        self._thread = None

        # Models/detectors are loaded once.
        self._facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        self._classifier = joblib.load(str(CLASSIFIER_PATH))
        self._label_encoder = joblib.load(str(LABEL_ENCODER_PATH))

        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self._haar.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade.")

        try:
            self._mtcnn = MTCNN(keep_all=True, device=DEVICE)
        except Exception:
            self._mtcnn = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._browser_frame = frame_bgr.copy()
            self._browser_frame_id += 1
            self._capture_error = None
            self._cond.notify_all()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def get_config(self):
        with self._lock:
            return dict(self._config)

    def update_config(self, payload):
        with self._lock:
            new_config = sanitize_update(payload, self._config)
            self._config = new_config

            # Reset processing state so changes apply immediately
            self._frame_index = 0
            self._cached_predictions = []

        return new_config

    def wait_for_frame(self, last_frame_id, timeout=1.0):
        with self._cond:
            if self._frame_id == last_frame_id:
                self._cond.wait(timeout=timeout)
            return self._frame_id, self._latest_jpeg

    def _open_capture_if_needed(self, config):
        desired = (config["camera_index"], config["frame_width"], config["frame_height"], config["capture_fps"])
        if self._cap is not None and self._cap_settings == desired and self._cap.isOpened():
            return

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        index = config["camera_index"]
        if index is None:
            index = find_camera_index()

        if index is None:
            self._capture_error = "No se pudo abrir la webcam (índice auto)."
            self._cap_settings = desired
            return

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self._capture_error = "No se pudo abrir la webcam (índice=%s)." % index
            self._cap_settings = desired
            try:
                cap.release()
            except Exception:
                pass
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["frame_width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["frame_height"]))
        cap.set(cv2.CAP_PROP_FPS, int(config["capture_fps"]))

        self._cap = cap
        self._cap_settings = desired
        self._capture_error = None

    def _detect_faces(self, frame_bgr, config):
        backend = config["face_detector"]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if backend == "mtcnn":
            if self._mtcnn is None:
                return [], None
            faces = detect_faces(self._mtcnn, frame_rgb, backend="mtcnn")
            return faces, frame_rgb

        # Haar
        faces = detect_faces(self._haar, frame_rgb, backend="haar")
        return faces, frame_rgb

    def _run(self):
        last_browser_frame_id = 0

        while not self._stop.is_set():
            with self._lock:
                config = dict(self._config)

                if self._browser_frame_id != last_browser_frame_id and self._browser_frame is not None:
                    frame = self._browser_frame.copy()
                    browser_frame_id = self._browser_frame_id
                else:
                    frame = None
                    browser_frame_id = last_browser_frame_id

            if frame is None:
                self._open_capture_if_needed(config)

                if self._cap is None or not self._cap.isOpened():
                    width = int(config["frame_width"])
                    height = int(config["frame_height"])
                    msg = self._capture_error or "Webcam no disponible."
                    frame = make_info_frame(msg, width, height)
                    time.sleep(0.1)
                    self._publish_frame(frame)
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    width = int(config["frame_width"])
                    height = int(config["frame_height"])
                    frame = make_info_frame("Error leyendo la webcam.", width, height)
                    time.sleep(0.05)
                    self._publish_frame(frame)
                    continue
            else:
                last_browser_frame_id = browser_frame_id

            self._frame_index += 1
            n = max(1, int(config["process_every_n_frames"]))
            do_process = ((self._frame_index - 1) % n == 0)

            img_h, img_w = frame.shape[:2]

            if do_process:
                self._cached_predictions = []

                faces, frame_rgb = self._detect_faces(frame, config)

                max_faces = int(config["max_faces"])
                if max_faces > 0 and len(faces) > max_faces:
                    faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)[:max_faces]

                for face in faces:
                    x, y, w, h = face.get("box", (0, 0, 0, 0))
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(img_w, x1 + max(0, int(w)))
                    y2 = min(img_h, y1 + max(0, int(h)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    if frame_rgb is not None:
                        face_pixels = frame_rgb[y1:y2, x1:x2]
                    else:
                        face_bgr = frame[y1:y2, x1:x2]
                        face_pixels = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                    embedding = get_embedding(self._facenet, DEVICE, face_pixels)
                    if embedding is None or len(embedding) == 0:
                        continue

                    emb = embedding.reshape(1, -1)
                    probabilities = self._classifier.predict_proba(emb)[0]
                    idx = int(np.argmax(probabilities))
                    label = self._label_encoder.inverse_transform([idx])[0]
                    conf = float(probabilities[idx])

                    self._cached_predictions.append((x1, y1, x2, y2, label, conf))

            # Draw cached predictions on every frame
            draw_color_rgb = get_color(config["draw_color"])
            draw_color_bgr = draw_color_rgb[::-1]  # OpenCV uses BGR
            draw_thick = int(config["draw_thickness"])
            for (x1, y1, x2, y2, label, conf) in self._cached_predictions:
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color_bgr, draw_thick)
                cv2.putText(
                    frame,
                    "%s (%.2f)" % (label, conf),
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    draw_color_bgr,
                    draw_thick,
                )

            # FPS overlay (display FPS)
            self._fps_count += 1
            dt = time.time() - self._fps_t0
            if dt >= 1.0:
                self._fps = self._fps_count / dt
                self._fps_count = 0
                self._fps_t0 = time.time()

            cv2.putText(
                frame,
                "FPS: %.2f" % self._fps,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                draw_color_bgr,
                draw_thick,
            )

            self._publish_frame(frame)

    def _publish_frame(self, frame_bgr):
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        with self._cond:
            self._latest_jpeg = jpg.tobytes()
            self._frame_id += 1
            self._cond.notify_all()


WORKER = CameraWorker()


def decode_frame_bytes(raw_bytes):
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    return frame


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, data, content_type="application/octet-stream", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            if not INDEX_HTML_PATH.exists():
                self._send_bytes(b"Missing app/index.html", "text/plain; charset=utf-8", status=500)
                return
            self._send_bytes(INDEX_HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return

        if self.path == "/style.css":
            if not STYLE_CSS_PATH.exists():
                self._send_bytes(b"Missing app/style.css", "text/plain; charset=utf-8", status=500)
                return
            self._send_bytes(STYLE_CSS_PATH.read_bytes(), "text/css; charset=utf-8")
            return

        if self.path == "/api/params":
            self._send_json(WORKER.get_config())
            return

        if self.path == "/video":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_id = 0
            try:
                while True:
                    fid, jpg = WORKER.wait_for_frame(last_id, timeout=1.0)
                    if jpg is None:
                        time.sleep(0.05)
                        continue

                    last_id = fid
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode("utf-8"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

        self._send_bytes(b"Not found", "text/plain; charset=utf-8", status=404)

    def do_POST(self):
        if self.path == "/api/frame":
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                self._send_json({"error": "Empty frame"}, status=400)
                return

            frame = decode_frame_bytes(raw)
            if frame is None:
                self._send_json({"error": "Invalid image data"}, status=400)
                return

            WORKER.submit_frame(frame)
            self._send_json({"ok": True})
            return

        if self.path == "/api/params":
            payload = self._read_json()
            if payload is None or not isinstance(payload, dict):
                self._send_json({"error": "Invalid JSON"}, status=400)
                return

            new_config = WORKER.update_config(payload)
            self._send_json(new_config)
            return

        self._send_bytes(b"Not found", "text/plain; charset=utf-8", status=404)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local webcam face-recognition web app (MJPEG)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    WORKER.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = "http://%s:%d/" % (display_host, args.port)
    print("Serving on %s" % url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        WORKER.stop()


if __name__ == "__main__":
    main()
