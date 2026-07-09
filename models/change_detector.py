import cv2
import numpy as np


def detect_changes(mask_before, mask_after):
    """
    Returns:
        gained : New pixels appearing in after image
        lost   : Pixels disappearing from before image
    """

    gained = np.logical_and(mask_after == 255, mask_before == 0)
    lost = np.logical_and(mask_before == 255, mask_after == 0)

    return gained, lost


def add_layer_overlay(image, gained, lost,
                      gain_color, loss_color,
                      alpha=0.35):
    """
    Draw one layer overlay with custom colors.
    """

    overlay = image.copy()

    overlay[gained] = gain_color
    overlay[lost] = loss_color

    return cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0
    )


def combine_overlays(image, overlays):
    """
    Combine multiple overlays together.
    """

    combined = image.copy()

    for overlay in overlays:
        combined = cv2.addWeighted(
            combined,
            1.0,
            overlay,
            1.0,
            0
        )

    return combined