from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import cv2
import json
import numpy as np

from models.vegetation import vegetation_mask
from models.water import water_mask
from models.change_detector import (
    detect_changes,
    add_layer_overlay,
    combine_overlays
)

app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    image1 = request.files["image1"]
    image2 = request.files["image2"]

    filename1 = secure_filename(image1.filename)
    filename2 = secure_filename(image2.filename)

    path1 = os.path.join(UPLOAD_FOLDER, filename1)
    path2 = os.path.join(UPLOAD_FOLDER, filename2)

    image1.save(path1)
    image2.save(path2)

    # ----------------------------
    # Load Images
    # ----------------------------

    reference = cv2.imread(path1)
    base_image = cv2.imread(path2)

    h = min(reference.shape[0], base_image.shape[0])
    w = min(reference.shape[1], base_image.shape[1])

    reference = cv2.resize(reference, (w, h))
    base_image = cv2.resize(base_image, (w, h))

    metadata = {}
    summary = []

    # =====================================
    # VEGETATION
    # =====================================

    veg1 = vegetation_mask(reference)
    veg2 = vegetation_mask(base_image)

    gained, lost = detect_changes(
        veg1,
        veg2
    )

    vegetation_overlay = add_layer_overlay(
        base_image.copy(),
        gained,
        lost,
        gain_color=(0, 255, 0),
        loss_color=(0, 0, 255),
        alpha=0.35
    )

    before_pct = round(
        np.mean(veg1 > 0) * 100,
        2
    )

    after_pct = round(
        np.mean(veg2 > 0) * 100,
        2
    )

    metadata["vegetation"] = {

        "before": before_pct,

        "after": after_pct,

        "change": round(
            after_pct - before_pct,
            2
        )

    }

    summary.append(

        f"Vegetation changed by {after_pct-before_pct:.2f}%."

    )
    # =====================================
    # WATER
    # =====================================

    water1 = water_mask(reference)
    water2 = water_mask(base_image)

    gained, lost = detect_changes(
        water1,
        water2
    )

    water_overlay = add_layer_overlay(
        base_image.copy(),
        gained,
        lost,
        gain_color=(255, 255, 0),   # Cyan
        loss_color=(255, 0, 255),   # Purple
        alpha=0.35
    )

    before_pct = round(
        np.mean(water1 > 0) * 100,
        2
    )

    after_pct = round(
        np.mean(water2 > 0) * 100,
        2
    )

    metadata["water"] = {

        "before": before_pct,

        "after": after_pct,

        "change": round(
            after_pct - before_pct,
            2
        )

    }

    summary.append(

        f"Water coverage changed by {after_pct-before_pct:.2f}%."

    )

    # =====================================
    # COMBINED OVERLAY
    # =====================================

    combined_overlay = combine_overlays(
        base_image.copy(),
        [
            vegetation_overlay,
            water_overlay
        ]
    )

    # =====================================
    # SAVE OUTPUTS
    # =====================================

    vegetation_path = os.path.join(
        OUTPUT_FOLDER,
        "vegetation_overlay.png"
    )

    water_path = os.path.join(
        OUTPUT_FOLDER,
        "water_overlay.png"
    )

    combined_path = os.path.join(
        OUTPUT_FOLDER,
        "combined_overlay.png"
    )

    cv2.imwrite(
        vegetation_path,
        vegetation_overlay
    )
    print("Vegetation saved:", os.path.exists(vegetation_path))

    cv2.imwrite(
        water_path,
        water_overlay
    )
    print("Water saved:", os.path.exists(water_path))

    cv2.imwrite(
        combined_path,
        combined_overlay
    )
    print("Combined saved:", os.path.exists(combined_path))

    with open(

        os.path.join(
            OUTPUT_FOLDER,
            "analysis.json"
        ),

        "w"

    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return render_template(
    "index.html",

    vegetation_overlay=vegetation_path.replace("\\", "/"),

    water_overlay=water_path.replace("\\", "/"),

    combined_overlay=combined_path.replace("\\", "/"),

    overlay=combined_path.replace("\\", "/"),

    image1=path1.replace("\\", "/"),

    image2=path2.replace("\\", "/"),

    metadata=metadata,

    summary=summary
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=True
    )