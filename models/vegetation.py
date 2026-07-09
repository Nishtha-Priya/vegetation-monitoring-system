import cv2
import numpy as np


def vegetation_mask(image):

    # Load image if a path is provided
    if isinstance(image, str):
        image = cv2.imread(image)

    # -----------------------------
    # CLAHE Contrast Enhancement
    # -----------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # -----------------------------
    # HSV Conversion
    # -----------------------------
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------
    # Green Detection
    # -----------------------------
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # -----------------------------
    # Morphology
    # -----------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -----------------------------
    # Remove Salt & Pepper Noise
    # -----------------------------
    mask = cv2.medianBlur(mask, 5)

    # -----------------------------
    # Remove Small Regions
    # -----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 350:
            clean_mask[labels == i] = 255

    return clean_mask