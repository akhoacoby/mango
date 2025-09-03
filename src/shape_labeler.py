import pandas as pd
import numpy as np


def classify_shape(eccentricity, solidity, aspect_ratio):
    if eccentricity > 0.85 and aspect_ratio > 1.4:
        return "elongated"
    elif solidity < 0.93:
        return "irregular"
    else:
        return "round"


def eccentricity(major, minor):
    if major <= 0 or minor <= 0:
        return 0.0
    major, minor = max(major, minor), min(major, minor)
    ratio = minor / major
    val = 1 - ratio**2
    if val < 0 or val > 1:
        print(
            f"[WARN] Invalid eccentricity: major={major}, minor={minor}, ratio={ratio}"
        )
        return 0.0
    return np.sqrt(val)


def solidity(area, convex_area):
    """Calculate solidity from area and convex area."""
    if convex_area == 0:
        return 0
    return area / convex_area


def aspect_ratio(bbox_width, bbox_height):
    if bbox_width <= 0 or bbox_height <= 0:
        print(f"[WARN] Invalid bounding box: width={bbox_width}, height={bbox_height}")
        return 0.0
    return bbox_width / bbox_height


# ----------------------------------------------------------------
def label_shapes_from_features(feature_csv_path):
    df = pd.read_csv(feature_csv_path)

    # Extract needed columns and drop rows with missing shape features
    shape_data = df[
        ["image", "shape_eccentricity", "shape_solidity", "shape_aspect_ratio"]
    ].dropna()
    shape_data.columns = ["image", "eccentricity", "solidity", "aspect_ratio"]

    # Apply classification rule to each row
    shape_data["shape_label"] = shape_data.apply(
        lambda row: classify_shape(
            row["eccentricity"], row["solidity"], row["aspect_ratio"]
        ),
        axis=1,
    )

    print("[INFO] Shape label distribution:")
    print(shape_data["shape_label"].value_counts())

    shape_df = shape_data[["image", "shape_label"]]

    return shape_df


if __name__ == "__main__":
    label_shapes_from_features("mango_features_new.csv")
