import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

import seaborn as sns
import matplotlib.pyplot as plt

h_bins = 30
s_bins = 32


# Define the path to the CSV file
csv_path = "mango_features.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)
# Display the first few rows of the DataFrame
# print(df.head())

# ---------------------------------------------------------


def color_label_from_avg_hsv(hue, sat):
    if 30 <= hue <= 110:
        return "fresh"
    elif 22 <= hue < 30:
        return "mid-ripe"
    elif 18 <= hue < 22:
        return "ripe"
    elif hue < 18:
        return "over-ripe"
    else:
        return "unknown"


# take the hsv features
hsv_df = df.loc[
    :, [h for h in df.columns if "hsv" in h and h != "hsv_0"]
]  # Exclude the first hsv column which is black color

# find the max hsv of each row
color_df = pd.DataFrame()

color_df[["image", "avg_h", "avg_s", "avg_v"]] = df[
    ["image", "avg_h", "avg_s", "avg_v"]
]

color_df["lbp_mean"] = df[[f"lbp_{i}" for i in range(10)]].mean(axis=1)
color_df["lbp_std"] = df[[f"lbp_{i}" for i in range(10)]].std(axis=1)

color_df["color_label"] = color_df.apply(
    lambda row: color_label_from_avg_hsv(row["avg_h"], row["avg_s"]), axis=1
)
print(color_df["color_label"].value_counts())
color_df.sort_values(by="image", axis=0, inplace=True)
color_df.to_csv("mango_color_features.csv", index=False)
print(color_df.head())

# color_map_reverse = {v: k for k, v in color_label.items()}
# color_df["color_class"] = color_df["color_label"].map(color_map_reverse)


# ---------------------------------------------------------
quality_labels = {
    0: "healthy",
    1: "bruised",
    2: "rotten",
}


# ----------------------------------------------------------
shape_label = {
    0: "round",
    1: "oval",
}


# -----------------------------------------------------------
# === Prepare Data for Training ===


# Extract features and labels
features = df.drop(columns=["image", "avg_h", "avg_s", "avg_v"]).values
labels = color_df["color_label"].tolist()  # Use color labels for classification

X = np.array(features)
y = np.array(labels)

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Compute class weights ===
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}


# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=100
)

# === Define MLP Model ===
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(y_categorical.shape[1], activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# === Train Model ===
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
)

# === Evaluate ===
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

print(classification_report(y_true_labels, y_pred_labels))

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="YlGnBu",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
