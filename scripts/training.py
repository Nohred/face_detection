"""Train the SVM classifier from the persistent embeddings CSV.

Preferred source (new schema):
  data/embeddings/embeddings.csv

Legacy fallback (old schema):
  data/embeddings.csv
"""

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import joblib
import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

new_embeddings_path = os.path.join(base_dir, "data", "embeddings", "embeddings.csv")
legacy_embeddings_path = os.path.join(base_dir, "data", "embeddings.csv")

if os.path.exists(new_embeddings_path):
	df = pd.read_csv(new_embeddings_path)
	if "name" not in df.columns:
		raise RuntimeError("Expected column 'name' in new embeddings CSV")
	embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
	if not embedding_cols:
		raise RuntimeError("Expected embedding_* columns in new embeddings CSV")

	X = df[embedding_cols].to_numpy(dtype=np.float32)
	y = df["name"].astype(str).to_numpy()
	print("Using embeddings:", new_embeddings_path)
	print("Class distribution:")
	print(df["name"].value_counts())
else:
	df = pd.read_csv(
		legacy_embeddings_path,
		converters={"embedding": lambda x: np.fromstring((x.strip()[1:-1] if x.strip().startswith("[") else x), sep=" ")},
	)
	X = np.array(df["embedding"].tolist(), dtype=np.float32)
	y = df["class"].astype(str).to_numpy()
	print("Using legacy embeddings:", legacy_embeddings_path)
	print("Class distribution:")
	print(df["class"].value_counts())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

if len(label_encoder.classes_) < 2:
	raise RuntimeError("Need at least 2 different classes to train")

classifier = SVC(kernel="linear", probability=True)
classifier.fit(X, y_encoded)


### Measure accuracy of the classifier
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=20)
# classifier = SVC(kernel='linear', probability=True)
# classifier.fit(X_train, y_train)
# accuracy = classifier.score(X_test, y_test)


# # Print confusion matrix
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()

# Save the trained classifier
models_dir = os.path.join(base_dir, "models")
legacy_models_dir = os.path.join(base_dir, "model")
os.makedirs(models_dir, exist_ok=True)

model_save_path = os.path.join(models_dir, "face_classifier.joblib")
joblib.dump(classifier, model_save_path)
print(f"Classifier saved to {model_save_path}")

label_encoder_save_path = os.path.join(models_dir, "label_encoder.joblib")
joblib.dump(label_encoder, label_encoder_save_path)
print(f"Label encoder saved to {label_encoder_save_path}")

# Best-effort: also update legacy location if it exists.
if os.path.isdir(legacy_models_dir):
	try:
		joblib.dump(classifier, os.path.join(legacy_models_dir, "face_classifier.joblib"))
		joblib.dump(label_encoder, os.path.join(legacy_models_dir, "label_encoder.joblib"))
		print(f"Legacy model/ updated at {legacy_models_dir}")
	except Exception:
		pass