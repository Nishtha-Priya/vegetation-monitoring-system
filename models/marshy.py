import cv2
import numpy as np


def marshy_mask(water_mask, vegetation_mask):

    kernel = np.ones((15,15), np.uint8)

    water = cv2.dilate(
        water_mask,
        kernel,
        iterations=2
    )

    vegetation = cv2.dilate(
        vegetation_mask,
        kernel,
        iterations=1
    )

    marshy = cv2.bitwise_and(
        water,
        vegetation
    )

    return marshy