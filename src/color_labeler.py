import pandas as pd
import cv2
import os

from labelerv2 import (
    bgy_mask_color,
    estimate_ripeness,
)


def label_color_from_csv(csv_path, image_root="."):
    df = pd.read_csv(csv_path)

    color_labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_root, row["image"])
        if not os.path.exists(image_path):
            print(f"⚠️ File not found: {image_path}")
            color_labels.append("unknown")
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image failed to load.")

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            result = bgy_mask_color(hsv_image=hsv)

            if result is None:
                raise ValueError("bgy_mask_color returned None.")
            black_mask, green_mask, yellow_mask = result

            label = estimate_ripeness(black_mask, green_mask, yellow_mask)
            color_labels.append(label)

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            color_labels.append("error")

    df["color_label"] = color_labels
    df_quality = df[["image", "color_label"]]

    return df_quality


if __name__ == "__main__":
    csv_path = "mango_features_new.csv"  # your CSV file with an 'image' column
    image_root = "mango\\dataset.v18i.tensorflow\\train"  # e.g., "mango/dataset.v18i.tensorflow/test"
    labeled_quality = label_color_from_csv(csv_path, image_root)
    print(labeled_quality["color_label"].value_counts())
    labeled_quality.to_csv("color_labels.csv", index=False)
