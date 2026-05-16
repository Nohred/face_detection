"""Extract embeddings from images/<person>/ and persist them to the new CSV schema.

Writes (and appends incrementally) to:
  data/embeddings/embeddings.csv

Each row contains metadata + embedding_0..embedding_N.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from functions import detect_faces, get_embedding
from embeddings_store import EmbeddingsStore, person_id_from_name, utc_now_iso


base_dir = Path(__file__).resolve().parent.parent
images_dir = base_dir / "images"
embeddings_csv = base_dir / "data" / "embeddings" / "embeddings.csv"

if not images_dir.exists():
    raise RuntimeError(f"Missing images directory: {images_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running MTCNN on device: {device}")

detector = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    keep_all=True,
    device=device,
    post_process=False,
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

embedding_dim = int(getattr(resnet, "embedding_size", 512) or 512)
store = EmbeddingsStore(embeddings_csv, embedding_dim=embedding_dim)
store.ensure_exists()
store.load_index()

rows_to_append: list[dict] = []
discarded = 0
processed = 0
skipped = 0

for person_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
    name = person_dir.name
    pid = person_id_from_name(name)

    image_paths = sorted(
        [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    for image_path in image_paths:
        processed += 1
        rel_image_path = image_path.relative_to(base_dir).as_posix()
        if store.already_processed(rel_image_path):
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            discarded += 1
            continue

        img_h, img_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_faces(detector, image_rgb, backend="mtcnn", min_confidence=0)
        if not faces:
            print(f"Discard (no face): {rel_image_path}")
            discarded += 1
            continue

        faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        x, y, w, h = faces[0]["box"]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, x1 + max(1, int(w)))
        y2 = min(img_h, y1 + max(1, int(h)))
        if x2 <= x1 or y2 <= y1:
            discarded += 1
            continue

        face_bgr = image_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        embedding = get_embedding(resnet, device, face_rgb)
        if embedding is None:
            print(f"Discard (no embedding): {rel_image_path}")
            discarded += 1
            continue

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(embedding) != embedding_dim:
            print(f"Discard (dim mismatch): {rel_image_path}")
            discarded += 1
            continue

        row = {
            "person_id": pid,
            "name": name,
            "image_path": rel_image_path,
            "created_at": utc_now_iso(),
        }
        for i, v in enumerate(embedding.tolist()):
            row[f"embedding_{i}"] = float(v)
        rows_to_append.append(row)

stats = store.append_rows(rows_to_append)
print(f"Processed images: {processed}")
print(f"Appended rows: {stats.get('appended', 0)}")
print(f"Skipped (already in CSV): {skipped + stats.get('skipped', 0)}")
print(f"Discarded: {discarded}")
print(f"Embeddings saved to: {embeddings_csv}")