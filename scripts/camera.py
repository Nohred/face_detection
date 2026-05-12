from functions import detect_faces, get_embedding, find_camera_index
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os
import cv2
import time




# Performance knobs (can be overridden via env vars)
FACE_DETECTOR_BACKEND = 'haar'  # mtcnn | haar
DETECTION_SCALE = 0.5  # only for mtcnn (0 < scale <= 1)
MIN_FACE_CONFIDENCE = 0.8  # only for mtcnn
MAX_FACES = 2
PROCESS_EVERY_N_FRAMES = 5 

# Load FaceNet model
base_dir = os.path.join(os.getcwd())
model_path = os.path.join(base_dir, 'model', 'facenet_keras.h5')
model = load_model(model_path, compile=False)

# Load face detector
if FACE_DETECTOR_BACKEND == 'haar':
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise RuntimeError('Failed to load OpenCV Haar cascade for face detection.')
else:
    from mtcnn.mtcnn import MTCNN
    FACE_DETECTOR_BACKEND = 'mtcnn'
    detector = MTCNN()

if FACE_DETECTOR_BACKEND == 'mtcnn':
    # Clamp to sane values to avoid accidental slowdowns or errors
    DETECTION_SCALE = max(0.1, min(1.0, float(DETECTION_SCALE)))
else:
    DETECTION_SCALE = 1.0

# Load the trained classifier
classifier_path = os.path.join(base_dir, 'model', 'face_classifier.joblib')
classifier = joblib.load(classifier_path)

# Load the label encoder
label_encoder_path = os.path.join(base_dir, 'model', 'label_encoder.joblib')
label_encoder = joblib.load(label_encoder_path)

#clear the console
os.system('clear')

print("Starting webcam face recognition. Press 'q' to quit.")

camera_index = find_camera_index()
if camera_index is None:
    print("No webcam could be opened. Available /dev/video devices on this machine may not match OpenCV indices.")
    print("Try setting CAMERA_INDEX to the correct index, for example:")
    print("CAMERA_INDEX=1 python scripts/camera.py")
    raise SystemExit(1)

print(f"Using camera index {camera_index}")
cap = cv2.VideoCapture(camera_index)
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
        if FACE_DETECTOR_BACKEND == 'mtcnn':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if DETECTION_SCALE < 1.0:
                small_rgb = cv2.resize(
                    frame_rgb,
                    (0, 0),
                    fx=DETECTION_SCALE,
                    fy=DETECTION_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )
                faces = detect_faces(detector, small_rgb, backend='mtcnn', min_confidence=MIN_FACE_CONFIDENCE)

                inv = 1.0 / DETECTION_SCALE
                for face in faces:
                    x, y, w, h = face.get('box', (0, 0, 0, 0))
                    face['box'] = [int(x * inv), int(y * inv), int(w * inv), int(h * inv)]
            else:
                faces = detect_faces(detector, frame_rgb, backend='mtcnn', min_confidence=MIN_FACE_CONFIDENCE)
        else:
            frame_rgb = None
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, frame_gray, backend='haar')

        if MAX_FACES > 0 and len(faces) > MAX_FACES:
            faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)[:MAX_FACES]

        for face in faces:
            x, y, width, height = face['box']

            # Clip bbox to image bounds (MTCNN can return negative coords)
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, x1 + max(0, int(width)))
            y2 = min(img_h, y1 + max(0, int(height)))
            if x2 <= x1 or y2 <= y1:
                continue

            if frame_rgb is not None:
                face_pixels = frame_rgb[y1:y2, x1:x2]
            else:
                face_bgr = frame[y1:y2, x1:x2]
                face_pixels = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            embedding = get_embedding(model, face_pixels)
            if embedding is None:
                continue
            embedding_reshaped = embedding.reshape(1, -1)

            probabilities = classifier.predict_proba(embedding_reshaped)[0]
            predicted_class_index = np.argmax(probabilities)
            predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence = float(probabilities[predicted_class_index])

            cached_predictions.append((x1, y1, x2, y2, predicted_class_label, confidence))

    for (x1, y1, x2, y2, predicted_class_label, confidence) in cached_predictions:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{predicted_class_label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    
    #Put real fps on the frame
    frame_count += 1
    elapsed_time = time.perf_counter() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.perf_counter()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #clear the console
        os.system('clear')
        break

cap.release()
cv2.destroyAllWindows()