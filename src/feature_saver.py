import os
import glob
import cv2
import numpy as np
import pandas as pd
import skimage as ski
from sklearn.preprocessing import StandardScaler
import logging
from multiprocessing import Pool

from feature_extraction import (
    extract_features,
)

from cropper import crop_mango_block, crop_mode


# ------------------ Image Processing & Saving ------------------ #
def process_image(path):
    try:
        image = crop_mango_block(path)
        if image is None:
            logging.warning(f"Failed to load image: {path}")
            return None
        # image = crop_mode(image, mode="contour")
        features = extract_features(image)
        return os.path.basename(path), features
    except Exception as e:
        logging.error(f"Error processing {path}: {e}")
        return None


def run_feature_extraction(input_dir, output_csv="mango_features_new.csv"):
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    logging.info(f"Found {len(image_paths)} images in {input_dir}.")

    with Pool() as pool:
        results = pool.map(process_image, image_paths)

    valid_results = [r for r in results if r is not None and r[1] is not None]

    if not valid_results:
        logging.error(
            "No valid features extracted. Check image preprocessing or feature logic."
        )
        return

    image_names, feature_list = zip(*valid_results)

    # Create column names
    h_bins, s_bins = 30, 32
    color_feature_names = [f"hsv_{i}" for i in range(h_bins * s_bins)] + [
        "avg_h",
        "avg_s",
        "avg_v",
    ]
    lbp_feature_names = [f"lbp_{i}" for i in range(10)]
    glcm_feature_names = [
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
    ]
    shape_feature_names = [
        "shape_area",
        "shape_perimeter",
        "shape_eccentricity",
        "shape_solidity",
        "shape_aspect_ratio",
    ]
    all_feature_names = (
        color_feature_names
        + lbp_feature_names
        + glcm_feature_names
        + shape_feature_names
    )

    feature_array = np.array(feature_list)

    features_df = pd.DataFrame(feature_array, columns=all_feature_names)
    features_df.insert(0, "image", image_names)
    features_df.to_csv(output_csv, index=False)
    logging.info(f"Saved features to {output_csv}")


if __name__ == "__main__":
    run_feature_extraction(input_dir="mango/dataset.v18i.tensorflow/train")
