from functions import detect_faces, get_embedding, get_color
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import numpy as np
import os
import cv2
import time
import tensorflow as tf

# Initialize camera
camera_index = 1
cap = cv2.VideoCapture(camera_index)

# Initialize GPU
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# Performance knobs (can be overridden via env vars)
FACE_DETECTOR_BACKEND = 'mtcnn'  # mtcnn | haar
MAX_FACES = 2
PROCESS_EVERY_N_FRAMES = 1 

# Color for drawing bounding boxes and labels
color = get_color("#EEFF00")  # Green for recognized faces
thickness = 1

# Load face detector
if FACE_DETECTOR_BACKEND == 'haar':
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    FACE_DETECTOR_BACKEND = 'mtcnn'
    detector = MTCNN(keep_all=True, device=device)


# Base directory
base_dir = os.path.join(os.getcwd())

# Load the trained classifier
classifier_path = os.path.join(base_dir, 'model', 'face_classifier.joblib')
classifier = joblib.load(classifier_path)

# Load the label encoder
label_encoder_path = os.path.join(base_dir, 'model', 'label_encoder.joblib')
label_encoder = joblib.load(label_encoder_path)

# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to(device) # Move model to device

#clear the console
os.system('clear')

print("Starting webcam face recognition. Press 'q' to quit.")


ret, frame = cap.read()
while not ret:
    wait_time = 2
    print(f"Failed to access webcam at index {camera_index}. Retrying in {wait_time} seconds...")
    cap.release()
    time.sleep(wait_time)
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Use MJPG codec for better performance on some webcams

# Count fps each second and print it on the frame
frame_count = 0
fps = 10
start_time = time.perf_counter()
frame_index = 0
cached_predictions = []  # (x1, y1, x2, y2, label, confidence)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        print("Failed to capture frame. Exiting.")
        break
    img_h, img_w = frame.shape[:2]
    frame_index += 1

    do_process = ((frame_index - 1) % PROCESS_EVERY_N_FRAMES == 0)
    if do_process:
        cached_predictions = []

        # Detect faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        faces = detect_faces(detector, frame_rgb, backend=FACE_DETECTOR_BACKEND)

        # Filter max faces
        if len(faces) > MAX_FACES:
            faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)[:MAX_FACES]

        # Get embeddings and predict classes
        
        for face in faces:
            x, y, w, h = face['box']
            x1 = max(0, int(x)); y1 = max(0, int(y))
            x2 = min(img_w, x1 + max(1, int(w))); y2 = min(img_h, y1 + max(1, int(h)))
            if x2 <= x1 or y2 <= y1:
                continue

            image_bgr = frame
            face_bgr = image_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(resnet, device, face_rgb)

            embedding_reshaped = embedding.reshape(1, -1)

            probabilities = classifier.predict_proba(embedding_reshaped)[0]
            predicted_class_index = np.argmax(probabilities)
            predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence = float(probabilities[predicted_class_index])

            cached_predictions.append((x1, y1, x2, y2, predicted_class_label, confidence))

    for (x1, y1, x2, y2, predicted_class_label, confidence) in cached_predictions:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            f"{predicted_class_label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            thickness
        )
    
    #Put real fps on the frame
    frame_count += 1
    elapsed_time = time.perf_counter() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.perf_counter()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #clear the console
        os.system('clear')
        break

cap.release()
cv2.destroyAllWindows()