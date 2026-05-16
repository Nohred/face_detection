import hashlib
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("app")


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.functions import detect_faces, get_embedding
from scripts.embeddings_store import EmbeddingsStore, normalize_name, person_id_from_name, utc_now_iso


STATIC_DIR = APP_DIR / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"
STYLE_CSS_PATH = STATIC_DIR / "style.css"

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ALIGNED_DIR = DATA_DIR / "aligned"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_CSV_PATH = EMBEDDINGS_DIR / "embeddings.csv"
LEGACY_EMBEDDINGS_CSV_PATH = DATA_DIR / "embeddings.csv"  # old schema (class, embedding)

MODELS_DIR = PROJECT_DIR / "models"
LEGACY_MODELS_DIR = PROJECT_DIR / "model"

CLASSIFIER_PATH = MODELS_DIR / "face_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
LEGACY_CLASSIFIER_PATH = LEGACY_MODELS_DIR / "face_classifier.joblib"
LEGACY_LABEL_ENCODER_PATH = LEGACY_MODELS_DIR / "label_encoder.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 160  # MTCNN output size


FACENET: InceptionResnetV1 | None = None
HAAR: cv2.CascadeClassifier | None = None
MTCNN_DETECTOR: MTCNN | None = None
EMBEDDING_DIM: int = 512

CLASSIFIER = None
LABEL_ENCODER = None
MODEL_LOCK = threading.Lock()

EMBEDDINGS_STORE: EmbeddingsStore | None = None

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


class InferenceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = {
            "face_detector": "haar",
            "max_faces": 5,
            "process_every_n_frames": 1,
            # Client-side drawing (kept so the UI can round-trip these defaults)
            "draw_color": "#00FF00",
            "draw_thickness": 2,
        }
        self.frame_index = 0
        self.cached_faces = []  # list[dict]


STATE = InferenceState()


def _ensure_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except Exception:
        return path.as_posix()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _try_copy_legacy_models() -> None:
    """Best-effort: if only legacy model/ exists, copy into models/ so docker volumes are simpler."""
    if CLASSIFIER_PATH.exists() and LABEL_ENCODER_PATH.exists():
        return
    if not (LEGACY_CLASSIFIER_PATH.exists() and LEGACY_LABEL_ENCODER_PATH.exists()):
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not CLASSIFIER_PATH.exists():
        shutil.copy2(LEGACY_CLASSIFIER_PATH, CLASSIFIER_PATH)
    if not LABEL_ENCODER_PATH.exists():
        shutil.copy2(LEGACY_LABEL_ENCODER_PATH, LABEL_ENCODER_PATH)


def _load_models_once() -> None:
    global FACENET, HAAR, MTCNN_DETECTOR, EMBEDDING_DIM

    if FACENET is None:
        logger.info("Loading FaceNet (InceptionResnetV1) on %s...", DEVICE)
        FACENET = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        EMBEDDING_DIM = int(getattr(FACENET, "embedding_size", 512) or 512)

    if HAAR is None:
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if haar.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade.")
        HAAR = haar

    if MTCNN_DETECTOR is None:
        try:
            MTCNN_DETECTOR = MTCNN(
                image_size=IMAGE_SIZE,
                margin=20,
                min_face_size=40,
                keep_all=True,
                device=DEVICE,
                post_process=False,
            )
        except Exception as e:
            logger.warning("Failed to initialize MTCNN detector: %s", e)
            MTCNN_DETECTOR = None


def _load_classifier(force: bool = False) -> dict:
    global CLASSIFIER, LABEL_ENCODER

    with MODEL_LOCK:
        if not force and CLASSIFIER is not None and LABEL_ENCODER is not None:
            return {"loaded": True, "source": "memory"}

        CLASSIFIER = None
        LABEL_ENCODER = None

        if CLASSIFIER_PATH.exists() and LABEL_ENCODER_PATH.exists():
            CLASSIFIER = joblib.load(str(CLASSIFIER_PATH))
            LABEL_ENCODER = joblib.load(str(LABEL_ENCODER_PATH))
            return {"loaded": True, "source": _relpath(MODELS_DIR)}

        if LEGACY_CLASSIFIER_PATH.exists() and LEGACY_LABEL_ENCODER_PATH.exists():
            CLASSIFIER = joblib.load(str(LEGACY_CLASSIFIER_PATH))
            LABEL_ENCODER = joblib.load(str(LEGACY_LABEL_ENCODER_PATH))
            return {"loaded": True, "source": _relpath(LEGACY_MODELS_DIR)}

        return {"loaded": False, "source": None}


def _best_effort_map_legacy_image_paths(class_name: str, expected_count: int) -> list[str] | None:
    """Try to recover image_path for the legacy (class, embedding) CSV without recomputing embeddings."""
    if MTCNN_DETECTOR is None:
        return None

    images_dir = PROJECT_DIR / "images" / class_name
    if not images_dir.exists() or not images_dir.is_dir():
        return None

    candidates = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )
    if not candidates:
        return None

    kept: list[str] = []
    for p in candidates:
        image_bgr = cv2.imread(str(p))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_faces(MTCNN_DETECTOR, image_rgb, backend="mtcnn", min_confidence=0)
        if faces:
            kept.append(_relpath(p))

    if len(kept) != int(expected_count):
        logger.warning(
            "Legacy image-path recovery mismatch for '%s': expected %s embeddings, found %s usable images",
            class_name,
            expected_count,
            len(kept),
        )
        return None
    return kept


def _migrate_legacy_embeddings_csv() -> dict:
    """Migrate data/embeddings.csv (old schema) -> data/embeddings/embeddings.csv (wide schema).

    Does NOT recompute embeddings. Tries to recover image_path from images/<class>/ if possible.
    """
    if EMBEDDINGS_CSV_PATH.exists() or not LEGACY_EMBEDDINGS_CSV_PATH.exists():
        return {"migrated": False}

    logger.info("Migrating legacy embeddings CSV '%s' -> '%s'", _relpath(LEGACY_EMBEDDINGS_CSV_PATH), _relpath(EMBEDDINGS_CSV_PATH))

    def _parse_embedding(value: str) -> np.ndarray:
        s = (value or "").strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        vec = np.fromstring(s, sep=" ")
        return vec

    df = pd.read_csv(str(LEGACY_EMBEDDINGS_CSV_PATH), converters={"embedding": _parse_embedding})
    if "class" not in df.columns or "embedding" not in df.columns:
        raise RuntimeError("Legacy embeddings.csv must contain columns: class, embedding")

    if df.empty:
        return {"migrated": False, "reason": "legacy_csv_empty"}

    # Determine embedding dim from legacy file (must match FaceNet for inference/training).
    dim = int(len(df.iloc[0]["embedding"]))
    if dim != EMBEDDING_DIM:
        logger.warning("Embedding dim mismatch: legacy=%s, facenet=%s. Proceeding with legacy dim.", dim, EMBEDDING_DIM)

    from scripts.embeddings_store import expected_header, embedding_columns
    header = expected_header(dim)
    cols = embedding_columns(dim)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()

    # Try to recover image paths per class.
    recovered_paths: dict[str, list[str]] = {}
    for class_name, sub in df.groupby("class"):
        paths = _best_effort_map_legacy_image_paths(str(class_name), expected_count=len(sub))
        if paths is not None:
            recovered_paths[str(class_name)] = paths

    with EMBEDDINGS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        # Write in original order to preserve correspondence with recovered_paths lists.
        class_offsets: dict[str, int] = {k: 0 for k in recovered_paths.keys()}
        for i, row in df.iterrows():
            name = str(row["class"])
            pid = person_id_from_name(name)

            image_path = f"legacy://{i}"
            if name in recovered_paths:
                j = class_offsets[name]
                if j < len(recovered_paths[name]):
                    image_path = recovered_paths[name][j]
                class_offsets[name] = j + 1

            embedding = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
            if len(embedding) != dim:
                raise RuntimeError(f"Inconsistent embedding length at row {i}: {len(embedding)} != {dim}")

            out = {
                "person_id": pid,
                "name": name,
                "image_path": image_path,
                "created_at": created_at,
            }
            for idx, c in enumerate(cols):
                out[c] = float(embedding[idx])
            writer.writerow(out)

    return {"migrated": True, "rows": int(len(df)), "embedding_dim": dim}


def decode_frame_bytes(raw_bytes):
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    return frame


def _detect_faces(frame_bgr, config):
    backend = config.get("face_detector", "haar")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if backend == "mtcnn":
        if MTCNN_DETECTOR is None:
            return [], frame_rgb
        faces = detect_faces(MTCNN_DETECTOR, frame_rgb, backend="mtcnn")
        return faces, frame_rgb

    faces = detect_faces(HAAR, frame_rgb, backend="haar")
    return faces, frame_rgb


def _infer_faces(frame_bgr, config):
    img_h, img_w = frame_bgr.shape[:2]
    faces, frame_rgb = _detect_faces(frame_bgr, config)

    max_faces = int(config.get("max_faces", 5) or 0)
    if max_faces > 0 and len(faces) > max_faces:
        faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)[:max_faces]

    results = []
    for face in faces:
        x, y, w, h = face.get("box", (0, 0, 0, 0))
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, x1 + max(0, int(w)))
        y2 = min(img_h, y1 + max(0, int(h)))
        if x2 <= x1 or y2 <= y1:
            continue

        face_rgb = frame_rgb[y1:y2, x1:x2]
        embedding = get_embedding(FACENET, DEVICE, face_rgb)
        if embedding is None or len(embedding) == 0:
            continue

        label = "unknown"
        conf = 0.0
        if CLASSIFIER is not None and LABEL_ENCODER is not None:
            emb = embedding.reshape(1, -1)
            probabilities = CLASSIFIER.predict_proba(emb)[0]
            idx = int(np.argmax(probabilities))
            label = str(LABEL_ENCODER.inverse_transform([idx])[0])
            conf = float(probabilities[idx])

        results.append(
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "label": str(label),
                "confidence": float(conf),
                "detector_confidence": float(face.get("confidence", 1.0)),
            }
        )

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_dirs()
    _load_models_once()
    _try_copy_legacy_models()

    # Embeddings CSV: ensure we can start from existing data without recomputing embeddings.
    migrated = _migrate_legacy_embeddings_csv()
    if migrated.get("migrated"):
        logger.info("Legacy embeddings migration completed: %s", migrated)

    global EMBEDDINGS_STORE
    EMBEDDINGS_STORE = EmbeddingsStore(EMBEDDINGS_CSV_PATH, embedding_dim=EMBEDDING_DIM)
    EMBEDDINGS_STORE.ensure_exists()
    EMBEDDINGS_STORE.load_index()

    # Classifier is optional at first boot.
    info = _load_classifier(force=True)
    if info.get("loaded"):
        logger.info("Classifier loaded (%s)", info.get("source"))
    else:
        logger.warning("No trained classifier found yet. Run /api/train after registering users.")

    yield


app = FastAPI(title="Face Recognition", version="3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="Missing app/static/index.html")
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/style.css")
def style():
    if not STYLE_CSS_PATH.exists():
        raise HTTPException(status_code=500, detail="Missing app/static/style.css")
    return FileResponse(path=str(STYLE_CSS_PATH), media_type="text/css")


@app.get("/api/status")
def api_status():
    store_rows = EMBEDDINGS_STORE.total_rows() if EMBEDDINGS_STORE is not None else 0
    users = EMBEDDINGS_STORE.users() if EMBEDDINGS_STORE is not None else []

    with MODEL_LOCK:
        classifier_loaded = CLASSIFIER is not None
        encoder_loaded = LABEL_ENCODER is not None

    return {
        "device": str(DEVICE),
        "embedding_dim": int(EMBEDDING_DIM),
        "mtcnn_available": bool(MTCNN_DETECTOR is not None),
        "classifier_loaded": bool(classifier_loaded),
        "label_encoder_loaded": bool(encoder_loaded),
        "embeddings_csv": _relpath(EMBEDDINGS_CSV_PATH),
        "embeddings_rows": int(store_rows),
        "users": [u.__dict__ for u in users],
    }


@app.get("/api/params")
def get_params():
    with STATE.lock:
        return dict(STATE.config)


@app.post("/api/params")
def update_params(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    with STATE.lock:
        STATE.config = sanitize_update(payload, STATE.config)
        STATE.frame_index = 0
        STATE.cached_faces = []
        return dict(STATE.config)


@app.post("/api/frame")
def process_frame(image_bytes: bytes = Body(..., media_type="image/jpeg")):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame")

    frame_bgr = decode_frame_bytes(image_bytes)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    with STATE.lock:
        config = dict(STATE.config)
        STATE.frame_index += 1
        current_index = int(STATE.frame_index)

        n = max(1, int(config.get("process_every_n_frames", 1) or 1))
        do_process = ((current_index - 1) % n == 0)
        if do_process:
            t0 = time.time()
            STATE.cached_faces = _infer_faces(frame_bgr, config)
            elapsed_ms = (time.time() - t0) * 1000.0
        else:
            elapsed_ms = 0.0

        img_h, img_w = frame_bgr.shape[:2]
        return {
            "frame_id": current_index,
            "width": int(img_w),
            "height": int(img_h),
            "faces": list(STATE.cached_faces),
            "processed": bool(do_process),
            "inference_ms": float(elapsed_ms),
        }


@app.get("/api/users")
def api_users():
    if EMBEDDINGS_STORE is None:
        return []
    return [u.__dict__ for u in EMBEDDINGS_STORE.users()]


@app.post("/api/register")
async def api_register(name: str = Form(...), images: list[UploadFile] = File(...)):
    if EMBEDDINGS_STORE is None:
        raise HTTPException(status_code=500, detail="Embeddings store not initialized")

    clean_name = normalize_name(name)
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    pid = person_id_from_name(clean_name)
    raw_person_dir = RAW_DIR / pid
    aligned_person_dir = ALIGNED_DIR / pid
    raw_person_dir.mkdir(parents=True, exist_ok=True)
    aligned_person_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows: list[dict] = []
    rejected: list[dict] = []
    skipped = 0
    received = 0
    seen_in_request: set[str] = set()

    # Prefer MTCNN for registration if available.
    detector_backend = "mtcnn" if MTCNN_DETECTOR is not None else "haar"

    for up in images:
        received += 1
        try:
            payload = await up.read()
        except Exception as e:
            rejected.append({"filename": up.filename, "reason": f"read_failed: {e}"})
            continue

        if not payload:
            rejected.append({"filename": up.filename, "reason": "empty_file"})
            continue

        digest = _sha256_hex(payload)
        aligned_path = aligned_person_dir / f"{digest}.jpg"
        aligned_rel = _relpath(aligned_path)
        if aligned_rel in seen_in_request or EMBEDDINGS_STORE.already_processed(aligned_rel):
            skipped += 1
            continue
        seen_in_request.add(aligned_rel)

        frame_bgr = decode_frame_bytes(payload)
        if frame_bgr is None:
            rejected.append({"filename": up.filename, "reason": "invalid_image"})
            continue

        raw_path = raw_person_dir / f"{digest}.jpg"
        if not raw_path.exists():
            try:
                cv2.imwrite(str(raw_path), frame_bgr)
            except Exception as e:
                rejected.append({"filename": up.filename, "reason": f"raw_save_failed: {e}"})
                continue

        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if detector_backend == "mtcnn":
            faces = detect_faces(MTCNN_DETECTOR, frame_rgb, backend="mtcnn")
        else:
            faces = detect_faces(HAAR, frame_rgb, backend="haar")

        if not faces:
            logger.info("Register discard: no_face filename=%s name=%s", up.filename, clean_name)
            rejected.append({"filename": up.filename, "reason": "no_face_detected"})
            continue

        # Choose largest detected face.
        faces = sorted(faces, key=lambda f: (f.get("box", [0, 0, 0, 0])[2] * f.get("box", [0, 0, 0, 0])[3]), reverse=True)
        x, y, w, h = faces[0].get("box", (0, 0, 0, 0))
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, x1 + max(0, int(w)))
        y2 = min(img_h, y1 + max(0, int(h)))
        if x2 <= x1 or y2 <= y1:
            rejected.append({"filename": up.filename, "reason": "invalid_face_box"})
            continue

        face_bgr = frame_bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            rejected.append({"filename": up.filename, "reason": "empty_face_crop"})
            continue

        try:
            cv2.imwrite(str(aligned_path), face_bgr)
        except Exception as e:
            rejected.append({"filename": up.filename, "reason": f"aligned_save_failed: {e}"})
            continue

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        embedding = get_embedding(FACENET, DEVICE, face_rgb)
        if embedding is None or len(embedding) == 0:
            logger.info("Register discard: no_embedding filename=%s name=%s", up.filename, clean_name)
            rejected.append({"filename": up.filename, "reason": "embedding_failed"})
            continue

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(embedding) != EMBEDDING_DIM:
            rejected.append({"filename": up.filename, "reason": f"embedding_dim_mismatch: {len(embedding)} != {EMBEDDING_DIM}"})
            continue

        row = {
            "person_id": pid,
            "name": clean_name,
            "image_path": aligned_rel,
            "created_at": utc_now_iso(),
        }
        for i, v in enumerate(embedding.tolist()):
            row[f"embedding_{i}"] = float(v)

        accepted_rows.append(row)

    write_stats = EMBEDDINGS_STORE.append_rows(accepted_rows)
    EMBEDDINGS_STORE.load_index()

    return {
        "person_id": pid,
        "name": clean_name,
        "received": int(received),
        "accepted": int(len(accepted_rows)),
        "persisted": int(write_stats.get("appended", 0)),
        "skipped": int(skipped + write_stats.get("skipped", 0)),
        "rejected": rejected,
        "embeddings_csv": _relpath(EMBEDDINGS_CSV_PATH),
    }


@app.post("/api/train")
def api_train():
    if not EMBEDDINGS_CSV_PATH.exists():
        raise HTTPException(status_code=400, detail="No embeddings CSV found. Register users first.")

    df = pd.read_csv(str(EMBEDDINGS_CSV_PATH))
    if df.empty:
        raise HTTPException(status_code=400, detail="Embeddings CSV is empty. Register users first.")
    if "name" not in df.columns:
        raise HTTPException(status_code=400, detail="Embeddings CSV missing 'name' column")

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    if not embedding_cols:
        raise HTTPException(status_code=400, detail="Embeddings CSV missing embedding_* columns")

    X = df[embedding_cols].to_numpy(dtype=np.float32)
    y = df["name"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    if len(encoder.classes_) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 different users to train a classifier")

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y_encoded)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(CLASSIFIER_PATH))
    joblib.dump(encoder, str(LABEL_ENCODER_PATH))

    # Keep legacy location up-to-date (optional, but helps existing scripts).
    if LEGACY_MODELS_DIR.exists():
        try:
            joblib.dump(clf, str(LEGACY_CLASSIFIER_PATH))
            joblib.dump(encoder, str(LEGACY_LABEL_ENCODER_PATH))
        except Exception:
            pass

    info = _load_classifier(force=True)
    return {
        "trained": True,
        "classes": list(encoder.classes_),
        "class_distribution": df["name"].value_counts().to_dict(),
        "classifier_path": _relpath(CLASSIFIER_PATH),
        "label_encoder_path": _relpath(LABEL_ENCODER_PATH),
        "reloaded": bool(info.get("loaded")),
    }


@app.post("/api/reload-model")
def api_reload_model():
    info = _load_classifier(force=True)
    if not info.get("loaded"):
        raise HTTPException(status_code=404, detail="No trained classifier found on disk")
    return info


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Local webcam face-recognition web app (FastAPI)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
