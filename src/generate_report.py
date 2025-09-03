import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


# -----------------------
# Helper: Smoothed curve
# -----------------------
def smooth_curve(data, weight=0.6):
    """Apply exponential smoothing to a list of values."""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# -----------------------
# Part 1: Load Training History
# -----------------------
with open("training_history.json", "r") as f:
    history = json.load(f)

# -----------------------
# Part 2: Load Test Data
# -----------------------
# Settings (customize if needed)
IMG_SIZE = 128
DATA_CSV = "mango_labeled_all.csv"
IMAGE_FOLDER = os.path.join("mango", "dataset.v18i.tensorflow", "train")

# Read CSV and filter out rows with missing labels
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

# Encode labels
from sklearn.preprocessing import LabelEncoder

le_color = LabelEncoder()
le_shape = LabelEncoder()
le_quality = LabelEncoder()

y_color = to_categorical(le_color.fit_transform(color_labels))
y_shape = to_categorical(le_shape.fit_transform(shape_labels))
y_quality = to_categorical(le_quality.fit_transform(quality_labels))

# Train/Test split (same as used originally)
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

# -----------------------
# Part 3: Load the Saved Model and Predict
# -----------------------
model = tf.keras.models.load_model("mango_model")
y_pred = model.predict(X_test)  # List: [pred_color, pred_shape, pred_quality]

# Get predicted classes (argmax) for each head
y_pred_color = np.argmax(y_pred[0], axis=1)
y_pred_shape = np.argmax(y_pred[1], axis=1)
y_pred_quality = np.argmax(y_pred[2], axis=1)

# True labels (argmax from one-hot)
y_true_color = np.argmax(y_color_test, axis=1)
y_true_shape = np.argmax(y_shape_test, axis=1)
y_true_quality = np.argmax(y_quality_test, axis=1)


# -----------------------
# Part 4: Compute Evaluation Metrics for Each Output Head
# -----------------------
def compute_metrics(y_true, y_pred, y_prob, head_name):
    # Compute precision, recall, f1 (using macro averaging)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    # Compute simulated mAP using average precision score. Convert true labels to one-hot.
    y_true_onehot = to_categorical(y_true, num_classes=y_prob.shape[1])
    mAP = average_precision_score(y_true_onehot, y_prob, average="macro")
    return precision, recall, f1, mAP


metrics_color = compute_metrics(y_true_color, y_pred_color, y_pred[0], "color")
metrics_shape = compute_metrics(y_true_shape, y_pred_shape, y_pred[1], "shape")
metrics_quality = compute_metrics(y_true_quality, y_pred_quality, y_pred[2], "quality")

# Compute confusion matrices
cm_color = confusion_matrix(y_true_color, y_pred_color)
cm_shape = confusion_matrix(y_true_shape, y_pred_shape)
cm_quality = confusion_matrix(y_true_quality, y_pred_quality)

# -----------------------
# Part 5: Plot the Complete Report
# -----------------------
# We'll create a 4 (rows) x 3 (columns) grid. Each column corresponds to a head.
# Row 1: Training Accuracy curves (raw + smoothed)
# Row 2: Training Loss curves (raw + smoothed)
# Row 3: Confusion Matrix (test set)
# Row 4: Text summary of metrics (Precision, Recall, F1, mAP)

output_heads = ["color", "shape", "quality"]

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
plt.suptitle("Mango Multi-Label CNN Training & Evaluation Report", fontsize=22)

# Row 1: Accuracy curves
for j, head in enumerate(output_heads):
    ax = axs[0, j]
    train_acc = history.get(f"{head}_accuracy", [])
    val_acc = history.get(f"val_{head}_accuracy", [])
    epochs = np.arange(len(train_acc))
    ax.plot(epochs, train_acc, label="Train", color="blue", marker="o")
    ax.plot(epochs, val_acc, label="Val", color="orange", marker="o")
    # Plot smoothed curves
    if len(train_acc) > 0:
        ax.plot(
            epochs,
            smooth_curve(train_acc),
            "--",
            color="blue",
            alpha=0.7,
            label="Train (smoothed)",
        )
        ax.plot(
            epochs,
            smooth_curve(val_acc),
            "--",
            color="orange",
            alpha=0.7,
            label="Val (smoothed)",
        )
    ax.set_title(f"{head.capitalize()} Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

# Row 2: Loss curves
for j, head in enumerate(output_heads):
    ax = axs[1, j]
    train_loss = history.get(f"{head}_loss", [])
    val_loss = history.get(f"val_{head}_loss", [])
    epochs = np.arange(len(train_loss))
    ax.plot(epochs, train_loss, label="Train", color="blue", marker="o")
    ax.plot(epochs, val_loss, label="Val", color="orange", marker="o")
    # Smoothed curves
    if len(train_loss) > 0:
        ax.plot(
            epochs,
            smooth_curve(train_loss),
            "--",
            color="blue",
            alpha=0.7,
            label="Train (smoothed)",
        )
        ax.plot(
            epochs,
            smooth_curve(val_loss),
            "--",
            color="orange",
            alpha=0.7,
            label="Val (smoothed)",
        )
    ax.set_title(f"{head.capitalize()} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

# # Row 3: Confusion Matrices
# cms = [cm_color, cm_shape, cm_quality]
# for j, head in enumerate(output_heads):
#     ax = axs[2, j]
#     cm = cms[j]
#     # Plot confusion matrix using seaborn heatmap
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#     ax.set_title(f"{head.capitalize()} Confusion Matrix")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")

# # Row 4: Metrics Summary (text)
# metrics_list = [metrics_color, metrics_shape, metrics_quality]
# for j, head in enumerate(output_heads):
#     ax = axs[3, j]
#     precision, recall, f1, mAP = metrics_list[j]
#     # Remove axes for text display.
#     ax.axis("off")
#     # Display metrics with some formatting
#     textstr = f"{head.capitalize()} Metrics:\n"
#     textstr += f"Precision: {precision:.3f}\n"
#     textstr += f"Recall:    {recall:.3f}\n"
#     textstr += f"F1 Score:  {f1:.3f}\n"
#     textstr += f"mAP@0.5:  {mAP:.3f}"
#     ax.text(
#         0.5,
#         0.5,
#         textstr,
#         fontsize=14,
#         ha="center",
#         va="center",
#         transform=ax.transAxes,
#         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
#     )

plt.tight_layout(rect=[0, 0, 1, 0.96])
report_path = "training_report.jpg"
plt.savefig(report_path, format="jpg", dpi=300)
plt.close()

print(f"[✓] Saved complete report image as {report_path}")
