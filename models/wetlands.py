import cv2
import numpy as np


def wetland_mask(vegetation_mask, water_mask):
    """
    Detect wetland areas by finding
    overlapping vegetation and water regions.
    """

    kernel = np.ones((5, 5), np.uint8)

    # slightly expand the masks
    vegetation = cv2.dilate(
        vegetation_mask,
        kernel,
        iterations=1
    )

    water = cv2.dilate(
        water_mask,
        kernel,
        iterations=1
    )

    wetland = cv2.bitwise_and(
        vegetation,
        water
    )

    return wetland