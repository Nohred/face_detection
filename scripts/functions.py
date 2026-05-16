import numpy as np
import cv2
import torch


### Detect faces in an image using MTCNN
def detect_faces(detector, image, backend="mtcnn", min_confidence=0):

    if backend == "mtcnn":
        # detector.detect from facenet-pytorch returns boxes and probabilities
        with torch.no_grad():
            boxes, probs = detector.detect(image)
        results = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < min_confidence:
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

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
def get_embedding(model, device, face_rgb):
    embeddings = [] 

    face_resized = cv2.resize(face_rgb, (160, 160)) # FaceNet expects 160x160 input
    face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float().to(device) # convert to CxHxW and float tensor
    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5  # normalize to [-1,1]

    with torch.no_grad(): # disable grad for inference
        emb = model(face_tensor.unsqueeze(0))
    embeddings.append(emb.cpu().numpy().reshape(-1))
    embeddings = np.vstack(embeddings) if embeddings else np.empty((0, model.embedding_size if hasattr(model, "embedding_size") else 512))
    return embeddings[0]

def get_color(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def find_camera_index():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    raise RuntimeError("No camera found")