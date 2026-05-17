# Face Detection and Recognition System

A web-based face detection and recognition application where the browser captures webcam frames, the backend (FastAPI) returns face detections as JSON, and the client draws bounding boxes on a JavaScript canvas.

## Project Structure

- `app/` - Web application files (server, HTML, CSS)
- `scripts/` - Utility scripts for training and data processing
- `model/` - Pre-trained models and classifiers
- `data/` - Embeddings and training data
- `images/` - Sample images for each person

## Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- A modern web browser with WebRTC support
- CUDA 11.x (optional, for GPU acceleration)

## Installation

### 1. Create the Conda Environment

Using the provided environment file:

```bash
conda env create -f environment.yml
conda activate facenet-env
```

### 2. Verify Model Files

Ensure the following files exist in the `model/` directory:
- `face_classifier.joblib` - Trained classifier for person identification
- `label_encoder.joblib` - Label encoder for class names

## Running the Server

The web application is running on
```
https://34.231.94.5.nip.io/
```

## Running Scripts

### Camera Live Detection

Run real-time face detection using your system webcam:

```bash
python scripts/camera.py
```

This script will:
- Detect faces in the video stream
- Extract face embeddings using FaceNet
- Classify detected faces using the trained classifier
- Display results with bounding boxes and confidence scores

Press 'q' to exit.

### Extract Face Embeddings

Generate embeddings from a set of face images:

```bash
cd scripts
python get_embeddings.py
```

This processes images in the `images/` directory and saves embeddings to `data/embeddings.csv`

### Train the Classifier

Train a new classifier using the extracted embeddings:

```bash
cd scripts
python training.py
```

This will generate new `face_classifier.joblib` and `label_encoder.joblib` files in the `model/` directory.

## Web Application Features

### Parameters

- **Detector**: Choose between 'haar' (fast) or 'mtcnn' (accurate) for face detection
- **Process Every N Frames**: Set detection frequency to balance speed and accuracy
- **Max Faces**: Limit the number of detected faces to process
- **Color**: Customize the color of detection bounding boxes (hex format)
- **Thickness**: Adjust the thickness of drawn rectangles and text (1-10)

### Interface

- The main display shows a canvas with the current frame and client-side bounding boxes
- Configuration panel allows real-time adjustment of parameters
- Status messages indicate system state and errors

## File Descriptions

### app/server.py

Main web server handling:
- HTTP requests for index page and styling
- Per-frame processing via individual HTTP requests
- Face detection and classification (returns JSON with bbox + name + probability)
- Configuration API endpoints

### scripts/functions.py

Core utility functions:
- `detect_faces()` - Face detection using Haar or MTCNN
- `get_embedding()` - Extract FaceNet embeddings
- `get_color()` - Convert hex colors to OpenCV BGR format

