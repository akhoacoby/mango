import numpy as np
from cropper import crop_mango_block
import cv2
from ultralytics import YOLO
from detect import detect_and_crop_mangoes

from skimage.feature import local_binary_pattern


def compute_texture_features(gray_img, mask=None):
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    if mask is not None:
        lbp_masked = lbp[mask == 255]
        variance = np.var(lbp_masked)
    else:
        variance = np.var(lbp)
    return variance


def color_uniformity_score(img, mask):
    masked_pixels = img[mask == 255]
    std_dev = np.std(masked_pixels, axis=0)  # BGR
    return np.mean(std_dev)  # average std across channels


def bgy_mask_color(hsv_image, mask=None):
    if hsv_image is None or hsv_image.ndim != 3 or hsv_image.shape[2] != 3:
        print("⚠️ Invalid HSV image passed to bgy_mask_color.")
        return None

    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    # Normalize HSV if needed (OpenCV uses 0–180 for H and 0–255 for S, V)
    if h.max() > 1.0:
        h = h / 180.0
        s = s / 255.0
        v = v / 255.0

    # Apply mask if provided
    if mask is not None:
        mask_bool = mask == 255
        h_valid = h[mask_bool]
        s_valid = s[mask_bool]
        v_valid = v[mask_bool]
    else:
        h_valid = h.flatten()
        s_valid = s.flatten()
        v_valid = v.flatten()

    if h_valid.size == 0:
        print("⚠️ Empty HSV valid region, skipping.")
        return None

    # black_mask = (v_valid < 0.3) & (s_valid < 0.3)
    black_mask = ((v_valid < 0.35) & (s_valid < 0.4)) | (
        (h_valid >= 0.05) & (h_valid <= 0.08) & (v_valid < 0.5) & (s_valid >= 0.4)
    )
    # green_mask = (h_valid >= 0.17) & (h_valid <= 0.4)
    green_mask = (
        (h_valid >= 0.17) & (h_valid <= 0.25) & (s_valid >= 0.4) & (v_valid >= 0.4)
    )

    yellow_mask = (h_valid >= 0.12) & (h_valid <= 0.17)
    # yellow_mask = (
    #     (h_valid >= 0.12) & (h_valid <= 0.17) & (s_valid >= 0.4) & (v_valid >= 0.5)
    # )

    return black_mask, green_mask, yellow_mask


def estimate_ripeness(black_mask, green_mask, yellow_mask):
    # Relative ratios (based on black+green+yellow only)
    black = np.sum(black_mask)
    green = np.sum(green_mask)
    yellow = np.sum(yellow_mask)

    total_color_pixels = black + green + yellow
    if total_color_pixels == 0:
        return "unknown"

    black_ratio = black / total_color_pixels
    green_ratio = green / total_color_pixels
    yellow_ratio = yellow / total_color_pixels

    print(
        f"🔹 Color proportions — Black: {black_ratio:.2f}, Green: {green_ratio:.2f}, Yellow: {yellow_ratio:.2f}"
    )

    if black_ratio >= 0.3:
        return "rotten"
    elif yellow_ratio > 0.7:
        return "ripe"
    elif green_ratio > 0.7:
        return "unripe"
    else:
        return "semi-ripe"


def estimate_quality(black_mask, mask):
    # Absolute black area ratio w.r.t full mango mask
    if mask is None or np.sum(mask == 255) == 0:
        return "unknown"

    mango_area = np.sum(mask == 255)
    black_area = np.sum(black_mask)

    black_ratio_absolute = black_area / mango_area

    print(f"🔹 Black area ratio over mango: {black_ratio_absolute:.2f}")

    if black_ratio_absolute >= 0.2:
        return "low quality"
    elif black_ratio_absolute <= 0.05:
        return "high quality"
    else:
        return "medium quality"


if __name__ == "__main__":
    img_path = r"E:\DATA SET\mango\test\images\1-13-_jpg.rf.06e9973d2139e1ceba6b581188cae1c0.jpg"

    crops = detect_and_crop_mangoes(img_path)

    if not crops:
        print("❌ Aucune mangue détectée dans l'image.")
        exit(1)

    for i, img in enumerate(crops):
        if img is None or img.size == 0:
            print(f"⚠️ Image {i} invalide ou vide, sautée.")
            continue

        hsv_crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # cv2.imshow(img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        black_mask, green_mask, yellow_mask = bgy_mask_color(hsv_crop, mask)

        if black_mask is None:
            print(f"⚠️ Masques non valides pour la mangue {i}")
            continue

        ripeness = estimate_ripeness(black_mask, green_mask, yellow_mask)
        quality = estimate_quality(black_mask, mask)

        print(f"\n🍋 Mango #{i + 1}")
        print(f"🔸 Ripeness: {ripeness}")
        print(f"🔸 Quality: {quality}")
