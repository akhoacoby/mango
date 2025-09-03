# cnn_classifier.py
import os
import numpy as np
import pandas as pd
import cv2
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# === Settings ===
IMG_SIZE = 128
DATA_CSV = "mango_labeled_all.csv"
IMAGE_FOLDER = "mango/dataset.v18i.tensorflow/train"


# ======================================
class MultiOutputDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y_dict, batch_size, datagen):
        self.X = X
        self.y_dict = y_dict
        self.batch_size = batch_size
        self.datagen = datagen
        self.indexes = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_X = self.X[batch_indexes]

        # Apply augmentation to each image
        batch_X_aug = np.array([self.datagen.random_transform(img) for img in batch_X])

        batch_y = {
            "color": self.y_dict["color"][batch_indexes],
            "shape": self.y_dict["shape"][batch_indexes],
            "quality": self.y_dict["quality"][batch_indexes],
        }

        return batch_X_aug, batch_y


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

X = np.array(images) / 255.0

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

# === Model definition ===
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3, 3), activation="relu")(inputs)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)

# Output heads
output_color = Dense(y_color.shape[1], activation="softmax", name="color")(x)
output_shape = Dense(y_shape.shape[1], activation="softmax", name="shape")(x)
output_quality = Dense(y_quality.shape[1], activation="softmax", name="quality")(x)

# Model
model = Model(inputs=inputs, outputs=[output_color, output_shape, output_quality])
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "color": "categorical_crossentropy",
        "shape": "categorical_crossentropy",
        "quality": "categorical_crossentropy",
    },
    metrics=["accuracy"],
)

model.summary()

# === Augmentation + Training ===
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

train_generator = MultiOutputDataGenerator(
    X_train,
    {"color": y_color_train, "shape": y_shape_train, "quality": y_quality_train},
    batch_size=32,
    datagen=datagen,
)

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=(
        X_test,
        {"color": y_color_test, "shape": y_shape_test, "quality": y_quality_test},
    ),
    verbose=1,
)


model.save("mango_model")

with open("training_history.json", "w") as f:
    json.dump(history.history, f)

# === YOLO-Style Training Report ===


def smooth_curve(data, weight=0.6):
    """Smoothing function for plots"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Get predictions on test set
y_pred = model.predict(X_test)
y_pred_color = np.argmax(y_pred[0], axis=1)
y_pred_shape = np.argmax(y_pred[1], axis=1)
y_pred_quality = np.argmax(y_pred[2], axis=1)

y_true_color = np.argmax(y_color_test, axis=1)
y_true_shape = np.argmax(y_shape_test, axis=1)
y_true_quality = np.argmax(y_quality_test, axis=1)


# Metric functions
def compute_prf(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return precision, recall, f1


def compute_map(y_true, y_score):
    y_true_bin = to_categorical(y_true)
    return average_precision_score(y_true_bin, y_score, average="macro")


# Compute metrics
color_p, color_r, color_f1 = compute_prf(y_true_color, y_pred_color)
shape_p, shape_r, shape_f1 = compute_prf(y_true_shape, y_pred_shape)
quality_p, quality_r, quality_f1 = compute_prf(y_true_quality, y_pred_quality)

map50_color = compute_map(y_true_color, y_pred[0])
map50_shape = compute_map(y_true_shape, y_pred[1])
map50_quality = compute_map(y_true_quality, y_pred[2])

fig, axs = plt.subplots(3, 4, figsize=(24, 12))
axs = axs.flatten()

# 6 training + val curves
plots = [
    ("color_loss", "val_color_loss"),
    ("shape_loss", "val_shape_loss"),
    ("quality_loss", "val_quality_loss"),
    ("color_accuracy", "val_color_accuracy"),
    ("shape_accuracy", "val_shape_accuracy"),
    ("quality_accuracy", "val_quality_accuracy"),
]

for i, (train_key, val_key) in enumerate(plots):
    train = history.history.get(train_key)
    val = history.history.get(val_key)
    if train is None or val is None:
        continue

    axs[i].plot(train, label="train", color="blue", marker=".")
    axs[i].plot(val, label="val", color="orange", marker=".")
    axs[i].plot(smooth_curve(train), "--", color="blue", alpha=0.5)
    axs[i].plot(smooth_curve(val), "--", color="orange", alpha=0.5)

    axs[i].set_title(train_key.replace("_", " ").title())
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel("Value")
    axs[i].legend()
    axs[i].grid(True)

# 6 bars: PRF + mAP
axs[6].bar(["P", "R", "F1"], [color_p, color_r, color_f1])
axs[6].set_ylim(0, 1)
axs[6].set_title("Color - Precision/Recall/F1")

axs[7].bar(["P", "R", "F1"], [shape_p, shape_r, shape_f1])
axs[7].set_ylim(0, 1)
axs[7].set_title("Shape - Precision/Recall/F1")

axs[8].bar(["P", "R", "F1"], [quality_p, quality_r, quality_f1])
axs[8].set_ylim(0, 1)
axs[8].set_title("Quality - Precision/Recall/F1")

axs[9].bar(["mAP@0.5"], [map50_color])
axs[9].set_ylim(0, 1)
axs[9].set_title("Color - mAP@0.5")

axs[10].bar(["mAP@0.5"], [map50_shape])
axs[10].set_ylim(0, 1)
axs[10].set_title("Shape - mAP@0.5")

axs[11].bar(["mAP@0.5"], [map50_quality])
axs[11].set_ylim(0, 1)
axs[11].set_title("Quality - mAP@0.5")

plt.tight_layout()
plt.savefig("training_report.jpg", format="jpg", dpi=300)
plt.show()


# === Inference on one image ===
def test_model(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img)
    pred_color = le_color.inverse_transform([np.argmax(preds[0])])[0]
    pred_shape = le_shape.inverse_transform([np.argmax(preds[1])])[0]
    pred_quality = le_quality.inverse_transform([np.argmax(preds[2])])[0]

    print(
        f"Prediction:\n - Color: {pred_color}\n - Shape: {pred_shape}\n - Quality: {pred_quality}"
    )


if __name__ == "__main__":
    test_image_path = "mango/dataset.v18i.tensorflow/test/sample.jpg"
    test_model(test_image_path)
