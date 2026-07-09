import cv2
import numpy as np


def align_images(path1, path2):
    """
    Align image2 to image1 using ORB feature matching.
    Falls back to resize if alignment fails.
    """

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError("Unable to read input images.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return resize_only(img1, img2, path1, path2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    matches = matches[:200]

    if len(matches) < 10:
        return resize_only(img1, img2, path1, path2)

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts2,
        pts1,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        return resize_only(img1, img2, path1, path2)

    h, w = img1.shape[:2]

    aligned = cv2.warpPerspective(
        img2,
        H,
        (w, h)
    )

    cv2.imwrite(path2, aligned)

    return path1, path2


def resize_only(img1, img2, path1, path2):

    h, w = img1.shape[:2]

    img2 = cv2.resize(
        img2,
        (w, h)
    )

    cv2.imwrite(path2, img2)

    return path1, path2