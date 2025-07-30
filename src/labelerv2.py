import numpy as np
from cropper import crop_mango_block
import cv2


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

    black_mask = v_valid < 0.5
    green_mask = (h_valid >= 0.17) & (h_valid <= 0.4)
    yellow_mask = (h_valid >= 0.10) & (h_valid <= 0.17)

    return black_mask, green_mask, yellow_mask


def estimate_ripeness(black_mask, green_mask, yellow_mask):
    total_color = np.sum(black_mask) + np.sum(green_mask) + np.sum(yellow_mask)

    black_ratio = np.sum(black_mask) / total_color
    green_ratio = np.sum(green_mask) / total_color
    yellow_ratio = np.sum(yellow_mask) / total_color

    if black_ratio > 0.3:
        return "rotten"
    elif yellow_ratio > 0.8:
        return "ripe"
    elif green_ratio > 0.8:
        return "unripe"
    else:
        return "semi-ripe"


def estimate_quality(black_mask, green_mask, yellow_mask):
    total_color = np.sum(black_mask) + np.sum(green_mask) + np.sum(yellow_mask)

    black_ratio = np.sum(black_mask) / total_color
    green_ratio = np.sum(green_mask) / total_color
    yellow_ratio = np.sum(yellow_mask) / total_color

    print(
        f"Black: {black_ratio:.2f}, Green: {green_ratio:.2f}, Yellow: {yellow_ratio:.2f}"
    )

    if black_ratio > 0.30:
        return "low quality"
    elif yellow_ratio > 0.65 or green_ratio > 0.65:
        return "high quality"
    else:
        return "medium quality"


if __name__ == "__main__":
    img_path = r"E:\DATA SET\mango\dataset.v18i.tensorflow\valid\1-23-_jpg.rf.418af36ea632b62a0d572606f538fb02.jpg"

    img = crop_mango_block(img_path)
    #     img, mask = crop_mode(img, mode="ellipse")

    # Preview (optional)
    cv2.imshow("Cropped Mango", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv_crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    black_mask, green_mask, yellow_mask = bgy_mask_color(hsv_image=hsv_crop, mask=None)
    print("Estimated quality:", estimate_quality(black_mask, green_mask, yellow_mask))

    print("Estimated ripeness:", estimate_ripeness(black_mask, green_mask, yellow_mask))
