import cv2
import numpy as np


def detect_changes(mask_before, mask_after):
    """
    Detect gained and lost regions between two masks.
    """

    gained = np.logical_and(mask_after == 255, mask_before == 0)
    lost = np.logical_and(mask_before == 255, mask_after == 0)

    return gained, lost


def add_layer_overlay(
    image,
    gained,
    lost,
    gain_color,
    loss_color,
    alpha=0.35
):
    """
    Creates colored overlay for a single layer.
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
    Blend multiple overlays together.
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


# ===================================================
# NEW FEATURE
# CHANGE HEATMAP
# ===================================================

def create_heatmap(mask_before, mask_after):
    """
    Create a smooth heatmap representing
    change intensity.
    """

    difference = cv2.absdiff(mask_before, mask_after)

    difference = cv2.GaussianBlur(
        difference,
        (31, 31),
        0
    )

    difference = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap = cv2.applyColorMap(
        difference.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.45):
    """
    Blend heatmap over original image.
    """

    return cv2.addWeighted(
        heatmap,
        alpha,
        image,
        1 - alpha,
        0
    )