import pandas as pd
import cv2
import os

from labelerv2 import (
    bgy_mask_color,
    estimate_quality,
)


def label_quality_from_csv(csv_path, image_root="."):
    df = pd.read_csv(csv_path)

    quality_labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_root, row["image"])
        if not os.path.exists(image_path):
            print(f"⚠️ File not found: {image_path}")
            quality_labels.append("unknown")
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

            label = estimate_quality(black_mask, green_mask, yellow_mask)
            quality_labels.append(label)

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            quality_labels.append("error")

    df["quality_label"] = quality_labels
    df_quality = df[["image", "quality_label"]]

    return df_quality


if __name__ == "__main__":
    csv_path = "mango_features_new.csv"  # your CSV file with an 'image' column
    image_root = "mango\\dataset.v18i.tensorflow\\train"  # e.g., "mango/dataset.v18i.tensorflow/test"
    labeled_quality = label_quality_from_csv(csv_path, image_root)
    print(labeled_quality["quality_label"].value_counts())
    labeled_quality.to_csv("quality_labels.csv", index=False)
