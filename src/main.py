# main.py
import argparse
import os
import logging

from color_labeler import label_color_from_csv
from shape_labeler import label_shapes_from_features
from quality_labeler import label_quality_from_csv
from feature_saver import run_feature_extraction


def combine_labels(
    feature_csv, color, shape, quality, output_csv="mango_labeled_all.csv"
):
    import pandas as pd

    features = pd.read_csv(feature_csv)

    merged = features.merge(color[["image", "color_label"]], on="image", how="left")
    merged = merged.merge(shape[["image", "shape_label"]], on="image", how="left")
    merged = merged.merge(quality[["image", "quality_label"]], on="image", how="left")

    merged.to_csv(output_csv, index=False)
    print(f"[✓] Combined labels saved to: {output_csv}")
    return merged


def run_mlp(final_csv_path):
    import train_classifier

    train_classifier.train(final_csv_path)


def run_cnn(final_csv_path):
    import cnn_classifier

    cnn_classifier.train(final_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--input_dir", default="mango/dataset.v18i.tensorflow/train")
    parser.add_argument("--output_csv", default="mango_labeled_all.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Feature extraction
    print("[1] Extracting features...")
    feature_csv = "mango_features_new.csv"
    run_feature_extraction(input_dir=args.input_dir, output_csv=feature_csv)

    # 2. Label each aspect
    print("[2] Labeling color...")
    color_df = label_color_from_csv(
        csv_path=feature_csv, image_root="mango/dataset.v18i.tensorflow/train"
    )

    print("[3] Labeling shape...")
    shape_df = label_shapes_from_features(feature_csv)

    print("[4] Labeling quality...")
    quality_df = label_quality_from_csv(
        csv_path=feature_csv, image_root="mango/dataset.v18i.tensorflow/train"
    )

    # 5. Combine labels
    print("[5] Combining all labels...")
    combine_labels(
        feature_csv=feature_csv,
        color=color_df,
        shape=shape_df,
        quality=quality_df,
        output_csv=args.output_csv,
    )

    # 6. Train model
    print(f"[6] Training model using: {args.mode.upper()}")
    if args.mode == "mlp":
        run_mlp(args.output_csv)
    elif args.mode == "cnn":
        run_cnn(args.output_csv)
