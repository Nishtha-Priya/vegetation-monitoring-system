import cv2
import numpy as np


def detect_changes(mask1, mask2):

    # Force both masks to the same size
    h = min(mask1.shape[0], mask2.shape[0])
    w = min(mask1.shape[1], mask2.shape[1])

    mask1 = cv2.resize(
        mask1,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    mask2 = cv2.resize(
        mask2,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    before = mask1 > 0
    after = mask2 > 0

    gained = (~before) & after
    lost = before & (~after)

    return gained, lost


def add_layer_overlay(
        image,
        gained,
        lost,
        gain_color,
        loss_color,
        alpha=0.55
):

    overlay = image.copy()

    h, w = image.shape[:2]

    gained = cv2.resize(
        gained.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    lost = cv2.resize(
        lost.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    color_layer = np.zeros_like(image)

    color_layer[gained] = gain_color
    color_layer[lost] = loss_color

    # Blend instead of replacing pixels
    result = cv2.addWeighted(
        image,
        1.0,
        color_layer,
        alpha,
        0
    )

    return result