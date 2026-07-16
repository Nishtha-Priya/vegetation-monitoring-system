import cv2
import numpy as np


def water_mask(image, use_texture_filter=False, texture_thresh=350.0, min_area=400):
    """
    Detect water bodies in an aerial/satellite RGB image.

    Pipeline:
      1. HSV color thresholding (tuned for blue/cyan water signatures)
      2. Morphological open/close to clean speckle and fill small gaps
      3. Median blur to remove salt & pepper noise
      4. Connected-component area filter (drop tiny blobs)
      5. (NEW) Texture/uniformity filter - drops blobs that are NOT
         spectrally smooth, which is what separates real water from
         greenish-blue wet fields, plastic mulch sheen, or crop canopy
         that leaks into the same hue range.

    Args:
        image: BGR image (np.ndarray) or path to an image file.
        use_texture_filter: if True, apply the smoothness check below.
            OFF by default - see warning below.
        texture_thresh: max allowed local-variance (grayscale) for a
            region to be kept as water. WARNING: real open water on
            large lakes/rivers with sediment plumes, wave patterns, or
            turbidity gradients commonly has grayscale variance in the
            300-450+ range - it is NOT spectrally flat at that scale.
            The default here (350) reflects that; a small/aggressive
            value like 10-20 will silently zero out your entire water
            mask. Calibrate this per-dataset by printing variance on
            known-good water blobs (see debug snippet in project notes)
            before trusting it, and treat it as risky for large/turbid
            water bodies rather than a general-purpose fix.
        min_area: minimum connected-component pixel area to keep.

    Returns:
        clean_mask: uint8 single-channel mask, 255 = water, 0 = not water.
    """

    if isinstance(image, str):
        image = cv2.imread(image)

    # ----------------------------
    # Convert to HSV
    # ----------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ----------------------------
    # Water Threshold
    # ----------------------------
    # Slightly narrowed from the original [80,60,50]-[130,255,220].
    # Raising the lower hue bound to 85 pulls the range away from the
    # green/teal zone where wet or sheen-covered vegetation tends to sit,
    # while still covering blue-to-violet-blue water tones.
    lower_blue = np.array([85, 60, 40])
    upper_blue = np.array([130, 255, 210])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ----------------------------
    # Morphological Operations
    # ----------------------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ----------------------------
    # Smooth Mask
    # ----------------------------
    mask = cv2.medianBlur(mask, 5)

    # ----------------------------
    # Remove Small Regions + Texture Filter
    # ----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    gray = None
    if use_texture_filter:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area <= min_area:
            continue

        if use_texture_filter:
            region_pixels = gray[labels == i]
            # Local variance of grayscale intensity inside the blob.
            # Water is spectrally flat -> low variance.
            # Crop rows / canopy / textured field surfaces -> higher variance.
            local_variance = float(np.var(region_pixels))

            if local_variance > texture_thresh:
                continue  # too textured to be water, likely a field

        clean_mask[labels == i] = 255

    return clean_mask