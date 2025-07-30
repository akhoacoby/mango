import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Set the random seed for reproducibility
np.random.seed(42)


# Crop the mango block from 1 image using the coordinates from the dataset
# it then will be used to extract features in the feature_saver.py
def crop_mango_block(image_path):
    """Crop a mango block from an image using bounding box coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    csv_path = Path(image_path).parent / "_annotations.csv"

    df = pd.read_csv(csv_path)

    image_row = df[df["filename"] == Path(image_path).name]

    if image_row.empty:
        logging.warning(
            f"No annotation found for: {image_path} → matched name: {Path(image_path).name}"
        )
        return None

    x_min = int(image_row["xmin"].iloc[0])
    y_min = int(image_row["ymin"].iloc[0])
    x_max = int(image_row["xmax"].iloc[0])
    y_max = int(image_row["ymax"].iloc[0])

    # Ensure bounding box is within image dimensions
    height, width = img.shape[:2]
    x_min = max(0, min(x_min, width - 1))
    x_max = max(1, min(x_max, width))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(1, min(y_max, height))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bounding box for image: {image_path}")

    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        raise ValueError(
            f"Empty crop at coordinates ({x_min}, {y_min}, {x_max}, {y_max}) for image: {image_path}"
        )

    return crop


def crop_mode(image, mode="contour"):
    """
    Crop the mango from the image using either bounding box, contour, or ellipse region.

    Args:
        image_path (str): Path to the input image.
        mode (str): One of "bbox", "contour", "ellipse". Determines cropping strategy.

    Returns:
        cropped_img (np.ndarray): The cropped image region, or None on failure.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # color range from black to yellow
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 50, 60])  # very low S and V
    color_lower = np.array([10, 40, 40])
    color_upper = np.array([40, 255, 255])
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_color = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.bitwise_or(mask_black, mask_color)

    # lower = np.array([0, 0, 0])
    # upper = np.array([40, 255, 255])
    # mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleaning
    kernel = np.ones((11, 11), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return None

    if mode == "bbox":
        return image[y : y + h, x : x + w], mask

    elif mode == "contour":
        mask_only = np.zeros_like(mask_clean)
        cv2.drawContours(mask_only, [contour], -1, 255, -1)
        result = cv2.bitwise_and(image, image, mask=mask_only)
        return result[y : y + h, x : x + w]  # , mask_only[y : y + h, x : x + w]

    elif mode == "ellipse":
        if len(contour) < 5:
            return None
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipse_mask = np.zeros_like(mask_clean)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            result = cv2.bitwise_and(image, image, mask=ellipse_mask)
            x1, y1, w1, h1 = cv2.boundingRect(ellipse_mask)
            return result[y1 : y1 + h1, x1 : x1 + w1], ellipse_mask[
                y1 : y1 + h1, x1 : x1 + w1
            ]
        except Exception as e:
            print(f"[ERROR] Ellipse fit failed: {e} in image:")
            return None

    else:
        print(f"[ERROR] Unknown crop mode: {mode}")
        return None


# test
if __name__ == "__main__":
    test_image_path = "mango\\dataset.v18i.tensorflow\\train\\-1_jpg.rf.0d7230e55a87aed1c5f2d00e37a0a2c4.jpg"
    try:
        cropped_image = crop_mango_block(test_image_path)
        print(f"Cropped image shape: {cropped_image.shape}")
        print("Cropped mango block saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Extract the mango blocks using coordinates in the dataset
# def extract_mango_blocks(dataset):
#     """Extract mango blocks from the dataset based on coordinates."""
#     mango_blocks = []
#     for index, row in dataset.iterrows():
#         x_min = int(row["xmin"])
#         y_min = int(row["ymin"])
#         x_max = int(row["xmax"])
#         y_max = int(row["ymax"])
#         mango_blocks.append((row["filename"], x_min, y_min, x_max, y_max))
#     return mango_blocks

#
# def crop_from_yolo_csv(
#     csv_path,
#     image_root,
#     output_dir,
#     return_saved_paths=False,
# ):
#     """Crop and save fruit blocks using bounding boxes from YOLO annotations."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     dataset = pd.read_csv(csv_path)
#     mango_blocks = extract_mango_blocks(dataset)
#     #     saved_paths = []
#     #     image_folder = image_root.split("\\")[-1]  # Get the last part of the path

#     for index, (filename, x_min, y_min, x_max, y_max) in enumerate(mango_blocks):
#         image_path = os.path.join(image_root, filename)
#         img = cv2.imread(image_path)

#         if img is None:
#             print(f"[!] Image not found: {filename}")
#             continue

#         # Ensure bounding box is within image dimensions
#         height, width = img.shape[:2]
#         x_min = max(0, min(x_min, width - 1))
#         x_max = max(1, min(x_max, width))
#         y_min = max(0, min(y_min, height - 1))
#         y_max = max(1, min(y_max, height))

#         if x_max <= x_min or y_max <= y_min:
#             print(f"[!] Skipping invalid box for image: {filename}")
#             continue

#         crop = img[y_min:y_max, x_min:x_max]
#         if crop.size == 0:
#             print(f"[!] Skipping empty crop at index {index} for {filename}")
#             continue

#      return crop

#         save_path = os.path.join(output_dir, f"{image_folder}_mango_block_{index}.jpg")
#         cv2.imwrite(save_path, crop)
#         if return_saved_paths:
#             saved_paths.append(save_path)

#     print(
#         f"[✓] Cropped {len(saved_paths) if return_saved_paths else len(mango_blocks)} mango blocks."
#     )
#     return saved_paths if return_saved_paths else None


# if __name__ == "__main__":
#     CSV_PATH = "mango/dataset.v18i.tensorflow/train/_annotations.csv"
#     TRAIN_IMG = "E:\\DATA SET\\mango\\dataset.v18i.tensorflow\\train"
#     VALID_IMG = "E:\\DATA SET\\mango\\dataset.v18i.tensorflow\\valid"
#     TEST_IMG = "E:\\DATA SET\\mango\\dataset.v18i.tensorflow\\test"
#     TRAIN_OUTPUT_DIR = "mango_blocks\\train"
#     VALID_OUTPUT_DIR = "mango_blocks\\valid"
#     TEST_OUTPUT_DIR = "mango_blocks\\test"

#     crop_from_yolo_csv(CSV_PATH, TRAIN_IMG, TRAIN_OUTPUT_DIR)
#     crop_from_yolo_csv(CSV_PATH, VALID_IMG, VALID_OUTPUT_DIR)
#     crop_from_yolo_csv(CSV_PATH, TEST_IMG, TEST_OUTPUT_DIR)
