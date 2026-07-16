document.addEventListener("DOMContentLoaded", () => {

    const overlayImage = document.getElementById("overlay-image");

    if (!overlayImage) return;

    const combined = document.getElementById("view-combined");
    const vegetation = document.getElementById("view-vegetation");
    const water = document.getElementById("view-water");
    const wetlands = document.getElementById("view-wetlands");
    const heatmap = document.getElementById("view-heatmap");

    const legends = {

        combined: document.getElementById("combined-legend"),

        vegetation: document.getElementById("vegetation-legend"),

        water: document.getElementById("water-legend"),

        wetlands:document.getElementById("wetlands-legend"),

        heatmap: document.getElementById("heatmap-legend")

    };

    function changeImage(layer){

        overlayImage.src = overlays[layer] + "?t=" + Date.now();

        Object.values(legends).forEach(legend => {

            if(legend){

                legend.style.display = "none";

            }

        });

        if(legends[layer]){

            legends[layer].style.display = "block";

        }

    }

    combined.addEventListener("change", () => {

        if(combined.checked){

            changeImage("combined");

        }

    });

    vegetation.addEventListener("change", () => {

        if(vegetation.checked){

            changeImage("vegetation");

        }

    });

    water.addEventListener("change", () => {

        if(water.checked){

            changeImage("water");

        }

    });

    wetlands.addEventListener("change",()=>{

        if(wetlands.checked){

            changeImage("wetlands");

        }

    });

    heatmap.addEventListener("change", () => {

        if(heatmap.checked){

            changeImage("heatmap");

        }

    });

    changeImage("combined");

});