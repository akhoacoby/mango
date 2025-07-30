from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import cv2
import os

# === Constants ===
IMG_SIZE = 128  # Change if different
IMAGE_FOLDER = "mango\\dataset.v18i.tensorflow\\valid"
DATA_CSV = "mango_labeled_all.csv"

# 1. Load the model
model = load_model("mango_model")

# === Load and preprocess data ===
df = pd.read_csv(DATA_CSV)
df.dropna(subset=["color_label", "shape_label", "quality_label"], inplace=True)

images = []
color_labels = []
shape_labels = []
quality_labels = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row["image"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    color_labels.append(row["color_label"])
    shape_labels.append(row["shape_label"])
    quality_labels.append(row["quality_label"])

X = np.array(images) / 255.0  # Normalize

# === Encode labels ===
le_color = LabelEncoder()
le_shape = LabelEncoder()
le_quality = LabelEncoder()

y_color = to_categorical(le_color.fit_transform(color_labels))
y_shape = to_categorical(le_shape.fit_transform(shape_labels))
y_quality = to_categorical(le_quality.fit_transform(quality_labels))

# === Train/Test split ===
(
    X_train,
    X_test,
    y_color_train,
    y_color_test,
    y_shape_train,
    y_shape_test,
    y_quality_train,
    y_quality_test,
) = train_test_split(X, y_color, y_shape, y_quality, test_size=0.2, random_state=42)

# 3. Run inference (prediction)
predictions = model.predict(X_test)

# If your model has multiple outputs:
pred_color, pred_shape, pred_quality = predictions

# 4. Evaluate the model
results = model.evaluate(
    X_test, [y_color_test, y_shape_test, y_quality_test], verbose=1
)

# Print all returned metrics with names
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")
