import tqdm
import pandas as pd
import cv2
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from functions import detect_faces, get_embedding

### Construct paths to the dataset and the face detector model
base_dir = os.path.join(os.getcwd())
images_dir = os.path.join(base_dir, 'images')

class_image_paths = {
    clase: [
        os.path.join(images_dir, clase, archivo)
        for archivo in sorted(os.listdir(os.path.join(images_dir, clase)))
        if archivo.lower().endswith(".jpg") or archivo.lower().endswith(".jpeg") or archivo.lower().endswith(".png")
    ]
    for clase in sorted(os.listdir(images_dir))
    if os.path.isdir(os.path.join(images_dir, clase))
}

# Determine default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running MTCNN on device: {device}")

# Load MTCNN and FaceNet models
detector = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to(device)
# Construct embeddings in a pandas DataFrame
data = []

# for clase, image_paths in class_image_paths.items():
for clase, image_paths in tqdm.tqdm(class_image_paths.items(), desc="Processing classes"):
    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        img_h, img_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_faces(detector, image_rgb)
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']

            x1 = max(0, int(x)); y1 = max(0, int(y))
            x2 = min(img_w, x1 + max(1, int(w))); y2 = min(img_h, y1 + max(1, int(h)))
            if x2 <= x1 or y2 <= y1:
                continue

            face_bgr = image_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(resnet, device, face_rgb)
            if embedding is None:
                print(f"Warning: No embedding extracted for {image_path}. Skipping.")
                continue
            data.append({'class': clase, 'embedding': embedding})

df = pd.DataFrame(data)
save_path = os.path.join(base_dir, 'data', 'embeddings.csv')
df.to_csv(save_path, index=False)
print(f"Embeddings saved to {save_path}")