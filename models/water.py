import cv2
import numpy as np


def water_mask(image):

    if isinstance(image, str):
        image = cv2.imread(image)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask