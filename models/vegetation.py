import cv2
import numpy as np


def vegetation_mask(image_path):

    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Improved green range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Remove small noise
    kernel = np.ones((7, 7), np.uint8)

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

    # Smooth edges
    mask = cv2.medianBlur(mask, 5)

    # Remove tiny regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 200:
            clean_mask[labels == i] = 255

    return clean_mask