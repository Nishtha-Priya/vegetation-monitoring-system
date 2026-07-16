import cv2
import numpy as np
from models.water import water_mask


def vegetation_mask(image, exclude_water=True):

    # Load image if a path is provided
    if isinstance(image, str):
        image = cv2.imread(image)

    original = image.copy()

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
    # Measured on two real captures of the same scene, taken years apart:
    # actual vegetation hue read as ~30-50 in one image and ~79-81 in the
    # other, due to differences in acquisition/color grading between
    # dates. A single narrow hue range can't cover both without also
    # catching something else, so this range is widened to ~25-85 to
    # span both clusters. That reopens overlap with teal/navy water
    # (H~90-97), so water is NOT excluded by hue here at all - it's
    # excluded explicitly below using the dedicated water detector,
    # which relies on its own tuned range + morphology and is not
    # fooled by vegetation's hue drift across dates.
    lower_green = np.array([25, 15, 10])
    upper_green = np.array([85, 200, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # -----------------------------
    # Explicitly Exclude Water
    # -----------------------------
    # Don't trust hue to keep water out - ask the water detector
    # directly and zero out anything it's confident is water.
    if exclude_water:
        wmask = water_mask(original)
        mask[wmask == 255] = 0

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