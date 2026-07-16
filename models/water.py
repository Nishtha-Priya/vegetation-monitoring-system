import cv2
import numpy as np


def water_mask(image):

    # ----------------------------
    # Convert to HSV
    # ----------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # ----------------------------
    # Water Threshold
    # ----------------------------

    lower_blue = np.array([85, 50, 40])
    upper_blue = np.array([140, 255, 255])


    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )


    # ----------------------------
    # Morphological Operations
    # ----------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
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


    # ----------------------------
    # Smooth Mask
    # ----------------------------

    mask = cv2.medianBlur(
        mask,
        5
    )


    # ----------------------------
    # Remove Small Regions
    # ----------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask
    )


    clean_mask = np.zeros_like(
        mask
    )


    MIN_AREA = 400


    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]


        if area > MIN_AREA:

            clean_mask[
                labels == i
            ] = 255


    return clean_mask