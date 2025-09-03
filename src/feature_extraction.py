import os
import numpy as np
import cv2
import pandas as pd
import skimage as ski
from sklearn.preprocessing import StandardScaler
import glob
import logging
from multiprocessing import Pool

from shape_labeler import eccentricity, solidity, aspect_ratio


# Suppress TensorFlow logs if imported elsewhere
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


# ------------------ Feature Extraction Functions ------------------ #
def extract_color_features(image, h_bins=30, s_bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    masked_pixels = hsv[mask > 0]
    avg_hue = np.mean(masked_pixels[:, 0]) if masked_pixels.size > 0 else 0
    avg_sat = np.mean(masked_pixels[:, 1]) if masked_pixels.size > 0 else 0
    avg_val = np.mean(masked_pixels[:, 2]) if masked_pixels.size > 0 else 0
    return hist, avg_hue, avg_sat, avg_val


def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = ski.feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist


def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = ski.feature.graycomatrix(
        gray, distances=[1], angles=[0], symmetric=True, normed=True
    )
    features = [
        ski.feature.graycoprops(glcm, prop)[0, 0]
        for prop in [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        ]
    ]
    return features


def extract_shape_features(image, show=False):
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 30, 50])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean internal defects
    kernel = np.ones((11, 11), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [0, 0, 0, 0, 0]

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Ellipse fit
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis <= 0 or minor_axis <= 0:
                raise ValueError("Invalid ellipse axes")
            cv2.ellipse(original, ellipse, (255, 0, 0), 2)
        except Exception as e:
            print(f"[ERROR] fitEllipse failed: {e}")
            major_axis = minor_axis = 0
    else:
        major_axis = minor_axis = 0

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        print(f"[WARN] Invalid bounding box: w={w}, h={h}")
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box

    hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(hull)

    ecc = eccentricity(major_axis, minor_axis)
    solid = solidity(area, convex_area)
    ar = aspect_ratio(w, h)

    if show:
        cv2.drawContours(original, [contour], -1, (0, 255, 0), 2)  # Green contour
        cv2.imshow("Contour, Ellipse and Bounding Box", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return [area, perimeter, ecc, solid, ar]


def extract_features(image):
    color_features, avg_h, avg_s, avg_v = extract_color_features(image)
    texture_features = extract_texture_features(image)
    glcm_features = extract_glcm_features(image)
    shape_features = extract_shape_features(image)
    return np.concatenate(
        [
            color_features,
            [avg_h, avg_s, avg_v],
            texture_features,
            glcm_features,
            shape_features,
        ]
    )


# test shape features extraction (contour drawing)
if __name__ == "__main__":
    test_image_path = "mango\\dataset.v18i.tensorflow\\train\\-3_jpg.rf.65668af7412fa57fb63e3000e833cca3.jpg"
    image = cv2.imread(test_image_path)
    if image is None:
        logging.error(f"Failed to load image: {test_image_path}")
    else:
        features = extract_shape_features(image)
        print(features)
