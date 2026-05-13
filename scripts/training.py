# Train a classifier (e.g., SVM) using the embeddings in the DataFrame
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import joblib
import numpy as np
import pandas as pd
import os

base_dir = os.path.join(os.getcwd())
embeddings_path = os.path.join(base_dir, 'data', 'embeddings.csv')
df = pd.read_csv(embeddings_path, converters={'embedding': lambda x: np.fromstring(x[1:-1], sep=' ')})

X = np.array(df['embedding'].tolist())
y = df['class'].values

# Print class distribution
print("Class distribution:")
print(df['class'].value_counts())


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

### Train with all data (no train-test split)
classifier = SVC(kernel='linear', probability=True)
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

model_save_path = os.path.join(base_dir, 'model', 'face_classifier.joblib')
joblib.dump(classifier, model_save_path)
print(f"Classifier saved to {model_save_path}")

# Save the label encoder
label_encoder_save_path = os.path.join(base_dir, 'model', 'label_encoder.joblib')
joblib.dump(label_encoder, label_encoder_save_path)
print(f"Label encoder saved to {label_encoder_save_path}")