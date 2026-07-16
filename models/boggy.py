import cv2
import numpy as np


def boggy_mask(water_mask, vegetation_mask):

    kernel = np.ones((25,25), np.uint8)

    water = cv2.dilate(
        water_mask,
        kernel,
        iterations=2
    )

    vegetation_inverse = cv2.bitwise_not(
        vegetation_mask
    )

    boggy = cv2.bitwise_and(
        water,
        vegetation_inverse
    )

    return boggy