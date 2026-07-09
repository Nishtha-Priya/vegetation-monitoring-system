document.addEventListener("DOMContentLoaded", function () {

    const vegetation = document.getElementById("toggle-vegetation");
    const water = document.getElementById("toggle-water");
    const overlayImage = document.getElementById("overlay-image");

    // Exit safely if elements don't exist
    if (!vegetation || !water || !overlayImage) {
        console.log("Layer controls not found.");
        return;
    }

    function updateOverlay() {

        let newImage = "";

        // Both selected
        if (vegetation.checked && water.checked) {

            newImage = overlays.combined;

        }

        // Vegetation only
        else if (vegetation.checked) {

            newImage = overlays.vegetation;

        }

        // Water only
        else if (water.checked) {

            newImage = overlays.water;

        }

        // Nothing selected
        else {

            overlayImage.style.display = "none";
            return;

        }

        overlayImage.style.display = "block";

        // Prevent browser caching
        overlayImage.src = newImage + "?t=" + Date.now();

    }

    vegetation.addEventListener("change", updateOverlay);
    water.addEventListener("change", updateOverlay);

    // Set correct overlay when page loads
    updateOverlay();

});