import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO


class MangoDetector:
    def __init__(
        self,
        classifier_path="mango_model",
        yolo_path="mango.pt",
        label_csv="mango_labeled_all.csv",
        img_size=128,
        conf_threshold=0.4,
    ):
        self.model = load_model(classifier_path)
        self.detector = YOLO(yolo_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        df = pd.read_csv(label_csv).dropna(
            subset=["color_label", "shape_label", "quality_label"]
        )
        self.le_color = LabelEncoder().fit(df["color_label"])
        self.le_shape = LabelEncoder().fit(df["shape_label"])
        self.le_quality = LabelEncoder().fit(df["quality_label"])

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        return np.expand_dims(img_norm, axis=0)

    def decode(self, preds):
        color, shape, quality = preds
        color = self.le_color.classes_[np.argmax(color)]
        shape = self.le_shape.classes_[np.argmax(shape)]
        quality = self.le_quality.classes_[np.argmax(quality)]
        return f"{color}, {shape}, {quality}"

    def __call__(self, image_input, save_path=None, show=False):
        # Accepts either image path or already-loaded image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input

        results = self.detector.predict(source=image, verbose=False)[0]

        for box in results.boxes:
            # print(f"Box: {box.xyxy[0]}, Confidence: {float(box.conf):.2f}")

            if float(box.conf) < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = self.preprocess(crop)
            preds = self.model.predict(input_tensor, verbose=0)
            label = self.decode(preds)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if save_path:
            cv2.imwrite(save_path, image)

        if show:
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image


detector_model = MangoDetector()

# # Option 1: pass path directly
# output_img = detector_model("your_image.jpg", save_path="out.jpg", show=True)

# Option 2: pass loaded image
img = cv2.imread(
    r"E:\DATA SET\mango\test\images\1-59-_jpg.rf.016b75405b8e104aa112bb4c45571f39.jpg"
)

detector_model(img, save_path="out.jpg", show=True)
