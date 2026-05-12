import pandas as pd
import cv2
import os
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from functions import detect_faces, get_embedding

### Construct paths to the dataset and the face detector model
base_dir = os.path.join(os.getcwd())
images_dir = os.path.join(base_dir, 'images')

class_image_paths = {
    clase: [
        os.path.join(images_dir, clase, archivo)
        for archivo in sorted(os.listdir(os.path.join(images_dir, clase)))
        if archivo.lower().endswith(".jpg") or archivo.lower().endswith(".jpeg")
    ]
    for clase in sorted(os.listdir(images_dir))
    if os.path.isdir(os.path.join(images_dir, clase))
}

# Load MTCNN and FaceNet models
detector = MTCNN()
model_path = os.path.join(base_dir, 'model', 'facenet_keras.h5')
model = load_model(model_path, compile=False)

# Construct embeddings in a pandas DataFrame
data = []

for clase, image_paths in class_image_paths.items():
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(detector, image_rgb)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']

            img_h, img_w = image_rgb.shape[:2]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, x1 + max(0, int(width)))
            y2 = min(img_h, y1 + max(0, int(height)))
            if x2 <= x1 or y2 <= y1:
                continue

            face_pixels = image_rgb[y1:y2, x1:x2]
            embedding = get_embedding(model, face_pixels)
            if embedding is None:
                continue
            data.append({'class': clase, 'embedding': embedding})

df = pd.DataFrame(data)
save_path = os.path.join(base_dir, 'data', 'embeddings.csv')
df.to_csv(save_path, index=False)
print(f"Embeddings saved to {save_path}")