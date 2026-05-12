import numpy as np
import cv2
import importlib
import os
import tensorflow as tf
from facenet_pytorch import MTCNN
import inspect
import torch


def _resolve_load_model():
    for module_name in ("keras.models", "tensorflow.keras.models"):
        try:
            module = importlib.import_module(module_name)
            return module.load_model
        except Exception:
            continue
    raise ImportError("Could not import a Keras load_model implementation.")


keras_load_model = _resolve_load_model()

def init_gpu():
    """Configure TensorFlow to use GPU with memory growth to avoid locking all VRAM.
    
    Falls back to CPU if CUDA compilation fails (e.g., cuDNN version mismatch).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detectada y configurada: {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(f"Error inicializando GPU: {e}")
            print("Usando CPU para inference")
    else:
        print("No se encontró GPU compatible con TensorFlow. Ejecutando en CPU.")

### Detect faces in an image using MTCNN
def detect_faces(detector, image, backend="mtcnn", min_confidence=None):
    backend = (backend or "mtcnn").lower().strip()

    if backend == "mtcnn":
        # detector.detect from facenet-pytorch returns boxes and probabilities
        boxes, probs = detector.detect(image)
        results = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if min_confidence is not None and prob < min_confidence:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box]
                width, height = x2 - x1, y2 - y1
                # Format to match OpenCV/MTCNN classic layout
                results.append({
                    'box': [max(0, x1), max(0, y1), width, height],
                    'confidence': prob
                })
        return results  # list of dicts with 'box' and 'confidence'

    if backend == "haar":
        # OpenCV Haar expects grayscale
        if image is None:
            return []
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        bboxes = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [
            {'box': [int(x), int(y), int(w), int(h)], 'confidence': 1.0}
            for (x, y, w, h) in bboxes
        ]

    raise ValueError(f"Unsupported backend '{backend}'. Use 'mtcnn' or 'haar'.")


def load_facenet_model(model_path):
    load_kwargs = {"compile": False}

    try:
        if "safe_mode" in inspect.signature(keras_load_model).parameters:
            load_kwargs["safe_mode"] = False
    except Exception:
        pass

    try:
        model = keras_load_model(model_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("safe_mode", None)
        model = keras_load_model(model_path, **load_kwargs)
    
    # Use eager execution to avoid CUDA graph compilation errors (e.g., cuDNN version mismatch)
    # This is slower but more compatible across CUDA environments
    model.run_eagerly = True
    
    return model

### Get embeddings from FaceNet
def get_embedding(model, face_pixels):
    if face_pixels is None or face_pixels.size == 0:
        return None

    # Resize first (significantly cheaper than resizing after expanding dims via TF)
    face_resized = cv2.resize(face_pixels, (160, 160), interpolation=cv2.INTER_LINEAR)
    face_resized = face_resized.astype('float32')

    # FaceNet-style prewhiten with std guard to avoid NaNs
    mean = face_resized.mean()
    std = face_resized.std()
    std_adj = max(std, 1.0 / np.sqrt(face_resized.size))
    face_whitened = (face_resized - mean) / std_adj

    samples = np.expand_dims(face_whitened, axis=0)
    yhat = model.predict_on_batch(samples)
    return yhat[0]


def find_camera_index(max_index=5):
    requested_index = os.environ.get('CAMERA_INDEX')
    if requested_index is not None:
        try:
            requested_index = int(requested_index)
        except ValueError:
            requested_index = None

    indices_to_try = []
    if requested_index is not None:
        indices_to_try.append(requested_index)
    indices_to_try.extend(idx for idx in range(max_index + 1) if idx not in indices_to_try)

    for index in indices_to_try:
        cap_test = cv2.VideoCapture(index)
        if cap_test.isOpened():
            cap_test.release()
            return index
        cap_test.release()

    return None