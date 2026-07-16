from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import cv2
import json
import numpy as np

from models.vegetation import vegetation_mask
from models.water import water_mask
from models.wetlands import wetland_mask
from models.marshy import marshy_mask
from models.boggy import boggy_mask
from models.change_detector import (
    detect_changes,
    add_layer_overlay,
    combine_overlays,
    create_heatmap,
    overlay_heatmap
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
        alpha=0.40
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
    # =====================================
    # WATER
    # =====================================

    water1 = water_mask(
        reference
    )

    water2 = water_mask(
        base_image
    )


    # DILATE WATER MASKS BEFORE CHANGE DETECTION

    kernel = np.ones(
        (7,7),
        np.uint8
    )

    water1 = cv2.dilate(
        water1,
        kernel,
        iterations=1
    )

    water2 = cv2.dilate(
        water2,
        kernel,
        iterations=1
    )


    water_gain, water_loss = detect_changes(

        water1,
        water2

    )
    water_gain, water_loss = detect_changes(
        water1,
        water2
    )

    # Create Water Overlay
    water_overlay = add_layer_overlay(

        base_image.copy(),

        water_gain,

        water_loss,

        gain_color=(255, 255, 0),

        loss_color=(255, 0, 255),

        alpha=0.40

    )


    # Water Metadata

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
    # =====================================
    # WETLANDS
    # =====================================

    wetland1 = wetland_mask(
        veg1,
        water1
    )

    wetland2 = wetland_mask(
        veg2,
        water2
    )

    gained, lost = detect_changes(
        wetland1,
        wetland2
    )

    wetland_overlay = add_layer_overlay(
        base_image.copy(),
        gained,
        lost,
        gain_color=(255, 255, 0),    # Cyan
        loss_color=(0, 165, 255),    # Orange
        alpha=0.40
    )

    before_pct = round(
        np.mean(wetland1 > 0) * 100,
        2
    )

    after_pct = round(
        np.mean(wetland2 > 0) * 100,
        2
    )

    metadata["wetlands"] = {

        "before": before_pct,

        "after": after_pct,

        "change": round(
            after_pct - before_pct,
            2
        )

    }

    # =====================================
    # MARSHY AREAS
    # =====================================

    marshy1 = marshy_mask(
        water1,
        veg1
    )

    marshy2 = marshy_mask(
        water2,
        veg2
    )

    gained, lost = detect_changes(
        marshy1,
        marshy2
    )

    marshy_overlay = add_layer_overlay(
        base_image.copy(),
        gained,
        lost,
        gain_color=(0,255,255),
        loss_color=(255,140,0),
        alpha=0.40
    )

    before_pct = round(
        np.mean(marshy1 > 0) * 100,
        2
    )

    after_pct = round(
        np.mean(marshy2 > 0) * 100,
        2
    )

    metadata["marshy"] = {

        "before": before_pct,

        "after": after_pct,

        "change": round(
            after_pct-before_pct,
            2
        )

    }

    # =====================================
    # BOGGY AREAS
    # =====================================

    boggy1 = boggy_mask(
        water1,
        veg1
    )

    boggy2 = boggy_mask(
        water2,
        veg2
    )

    gained, lost = detect_changes(
        boggy1,
        boggy2
    )

    boggy_overlay = add_layer_overlay(
        base_image.copy(),
        gained,
        lost,
        gain_color=(139,69,19),
        loss_color=(255,215,0),
        alpha=0.40
    )

    before_pct = round(
        np.mean(boggy1 > 0) * 100,
        2
    )

    after_pct = round(
        np.mean(boggy2 > 0) * 100,
        2
    )

    metadata["boggy"] = {

        "before": before_pct,

        "after": after_pct,

        "change": round(
            after_pct-before_pct,
            2
        )

    }

    # =====================================
    # AI ANALYSIS
    # =====================================

    summary = []

    veg_change = metadata["vegetation"]["change"]
    water_change = metadata["water"]["change"]

    # Vegetation
    if veg_change > 5:
        summary.append(
            f"🌿 Vegetation increased by {veg_change:.2f}%."
        )
    elif veg_change < -5:
        summary.append(
            f"🌿 Vegetation decreased by {abs(veg_change):.2f}%."
        )
    else:
        summary.append(
            "🌿 Vegetation remained relatively stable."
        )

    # Water
    if water_change > 3:
        summary.append(
            f"💧 Water bodies expanded by {water_change:.2f}%."
        )
    elif water_change < -3:
        summary.append(
            f"💧 Water bodies reduced by {abs(water_change):.2f}%."
        )
    else:
        summary.append(
            "💧 No significant water change detected."
        )
    # Wetlands
    wetland_change = metadata["wetlands"]["change"]

    if wetland_change > 2:

        summary.append(
            f"🌾 Wetland regions expanded by {wetland_change:.2f}%."
        )

    elif wetland_change < -2:

        summary.append(
            f"🌾 Wetland regions reduced by {abs(wetland_change):.2f}%."
        )

    else:

        summary.append(
            "🌾 Wetland coverage remained relatively stable."
        )
    # Marshy
    marshy_change = metadata["marshy"]["change"]
    if abs(marshy_change) > 2:

        summary.append(
            f"🌱 Marshy terrain changed by {abs(marshy_change):.2f}%."
        )

    # Boggy
    boggy_change = metadata["boggy"]["change"]
    if abs(boggy_change) > 2:

        summary.append(

            f"🟤 Boggy terrain changed by {abs(boggy_change):.2f}%."
        )

    # Terrain
    veg_after = metadata["vegetation"]["after"]
    water_after = metadata["water"]["after"]

    if veg_after > 60:
        terrain = "Dense Vegetation"

    elif veg_after > 35:
        terrain = "Agricultural / Mixed Vegetation"

    elif water_after > 20:
        terrain = "Water Dominated"

    else:
        terrain = "Sparse Vegetation / Bare Terrain"

    summary.append(f"🛰 Dominant Terrain: {terrain}")

    # Overall Change
    overall = abs(veg_change) + abs(water_change)

    if overall < 5:
        severity = "Low"

    elif overall < 15:
        severity = "Moderate"

    else:
        severity = "High"

    summary.append(
        f"📊 Overall Change Intensity: {severity}"
    )

    # =====================================
    # COMBINED OVERLAY
    # =====================================

    combined_overlay = combine_overlays(

        base_image.copy(),

        [
            vegetation_overlay,
            water_overlay,
            wetland_overlay,
            marshy_overlay,
            boggy_overlay
        ]
    )

    # =====================================
    # CHANGE HEATMAP
    # =====================================

    combined_before = cv2.bitwise_or(
        veg1,
        water1
    )

    combined_before = cv2.bitwise_or(
        combined_before,
        wetland1
    )

    combined_before = cv2.bitwise_or(
        combined_before,
        marshy1
    )

    combined_before = cv2.bitwise_or(
        combined_before,
        boggy1
    )


    combined_after = cv2.bitwise_or(
        veg2,
        water2
    )

    combined_after = cv2.bitwise_or(
        combined_after,
        wetland2
    )

    combined_after = cv2.bitwise_or(
        combined_after,
        marshy2
    )

    combined_after = cv2.bitwise_or(
        combined_after,
        boggy2
    )

    heatmap = create_heatmap(combined_before, combined_after)

    heatmap_overlay = overlay_heatmap(
        base_image.copy(),
        heatmap
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

    wetland_path = os.path.join(
        OUTPUT_FOLDER,
        "wetland_overlay.png"
    )

    marshy_path = os.path.join(
        OUTPUT_FOLDER,
        "marshy_overlay.png"
    )

    boggy_path = os.path.join(
        OUTPUT_FOLDER,
        "boggy_overlay.png"
    )

    combined_path = os.path.join(
        OUTPUT_FOLDER,
        "combined_overlay.png"
    )

    heatmap_path = os.path.join(
        OUTPUT_FOLDER,
        "change_heatmap.png"
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
        wetland_path,
        wetland_overlay
    )
    print("Wetlands saved:", os.path.exists(wetland_path))

    cv2.imwrite(
        marshy_path,
        marshy_overlay
    )
    print("Marshy saved:", os.path.exists(marshy_path))

    cv2.imwrite(
        boggy_path,
        boggy_overlay
    )
    print("Boggy saved:", os.path.exists(boggy_path))

    cv2.imwrite(
        combined_path,
        combined_overlay
    )
    print("Combined saved:", os.path.exists(combined_path))

    cv2.imwrite(
        heatmap_path,
        heatmap_overlay
    )

    print("Heatmap saved:", os.path.exists(heatmap_path))

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
    # =====================================
    # TERRAIN ESTIMATION
    # =====================================

    veg = metadata["vegetation"]["after"]
    water = metadata["water"]["after"]

    terrain = {}

    # Dominant terrain
    if water > 30:
        terrain["dominant"] = "Water Body"

    elif veg > 70:
        terrain["dominant"] = "Dense Vegetation"

    elif veg > 40:
        terrain["dominant"] = "Agricultural / Mixed Vegetation"

    elif veg > 15:
        terrain["dominant"] = "Sparse Vegetation"

    else:
        terrain["dominant"] = "Bare Terrain"

    # Vegetation Density
    if veg > 70:
        terrain["vegetation_density"] = "High"
    elif veg > 40:
        terrain["vegetation_density"] = "Moderate"
    elif veg > 15:
        terrain["vegetation_density"] = "Low"
    else:
        terrain["vegetation_density"] = "Very Low"

    # Water Presence
    if water > 20:
        terrain["water_presence"] = "High"
    elif water > 8:
        terrain["water_presence"] = "Moderate"
    else:
        terrain["water_presence"] = "Low"

    # Landscape Type

    if metadata["wetlands"]["after"] > 5:

        terrain["landscape"] = "Wetlands"

    elif metadata["marshy"]["after"] > 5:

        terrain["landscape"] = "Marshy Terrain"

    elif metadata["boggy"]["after"] > 5:

        terrain["landscape"] = "Boggy Terrain"

    elif water > 20 and veg > 40:

        terrain["landscape"] = "Mixed Wetland Terrain"

    elif water < 5 and veg < 20:

        terrain["landscape"] = "Dry / Bare Landscape"

    elif veg > 60:

        terrain["landscape"] = "Natural Vegetation"

    else:

        terrain["landscape"] = "Mixed Natural Terrain"
    metadata["terrain"] = terrain

    # =====================================
    # ROAD ACCESSIBILITY ANALYSIS
    # =====================================


    road_analysis = {}


    if metadata["wetlands"]["after"] > 5:

        road_analysis["roads_nearby"] = "Likely"

        road_analysis["accessibility"] = "High"


    elif metadata["marshy"]["after"] > 5:

        road_analysis["roads_nearby"] = "Possible"

        road_analysis["accessibility"] = "Moderate"


    elif metadata["boggy"]["after"] > 5:

        road_analysis["roads_nearby"] = "Uncertain"

        road_analysis["accessibility"] = "Low"


    else:

        road_analysis["roads_nearby"] = "Unknown"

        road_analysis["accessibility"] = "Unknown"


    metadata["road_analysis"] = road_analysis

    print("\nMARSHY:")
    print(metadata["marshy"])

    print("\nBOGGY:")
    print(metadata["boggy"])

    print("\nROAD ANALYSIS:")
    print(metadata["road_analysis"])

    print("\nTERRAIN:")
    print(metadata["terrain"])
    
    return render_template(
        "index.html",

        vegetation_overlay=vegetation_path.replace("\\", "/"),

        water_overlay=water_path.replace("\\", "/"),

        wetland_overlay=wetland_path.replace("\\", "/"),

        marshy_overlay=marshy_path.replace("\\", "/"),

        boggy_overlay=boggy_path.replace("\\", "/"),

        combined_overlay=combined_path.replace("\\", "/"),

        heatmap_overlay=heatmap_path.replace("\\", "/"),

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