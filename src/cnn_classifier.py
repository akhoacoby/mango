# cnn_classifier.py
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


# === Evaluate ===
eval_results = model.evaluate(
    X_test, {"color": y_color_test, "shape": y_shape_test, "quality": y_quality_test}
)
print(f"[✓] Test Losses and Accuracies: {dict(zip(model.metrics_names, eval_results))}")

# === Plotting ===
plt.plot(history.history["color_accuracy"], label="color acc")
plt.plot(history.history["val_color_accuracy"], label="val color acc")
plt.plot(history.history["shape_accuracy"], label="shape acc")
plt.plot(history.history["val_shape_accuracy"], label="val shape acc")
plt.plot(history.history["quality_accuracy"], label="quality acc")
plt.plot(history.history["val_quality_accuracy"], label="val quality acc")
plt.title("Multi-task Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
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
