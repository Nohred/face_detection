import numpy as np
import cv2
import os


### Detect faces in an image using MTCNN
def detect_faces(detector, image, backend="mtcnn", min_confidence=None):
    backend = (backend or "mtcnn").lower().strip()

    if backend == "mtcnn":
        results = detector.detect_faces(image)
        if min_confidence is not None:
            results = [r for r in results if r.get('confidence', 0.0) >= float(min_confidence)]
        return results  # bbox, confidence, keypoints

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