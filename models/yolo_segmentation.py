from ultralytics import YOLO
import cv2
import numpy as np

print("Loading YOLO11 Segmentation...")

# Downloads automatically on first run
model = YOLO("yolo11n-seg.pt")

print("YOLO Loaded!")


def segment_image(image_path):

    image = cv2.imread(image_path)

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
        retina_masks=True
    )

    return results[0]