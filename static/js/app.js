document.addEventListener("DOMContentLoaded", function () {

    // ==============================
    // Existing Toggle Functionality
    // ==============================

    function connect(checkId, sectionId) {

        const check = document.getElementById(checkId);
        const section = document.getElementById(sectionId);

        if (!check || !section) return;

        check.addEventListener("change", function () {

            section.style.display = check.checked ? "block" : "none";

        });
    }

    connect("toggle-before", "before-section");
    connect("toggle-after", "after-section");
    connect("toggle-change", "overlay-section");



    // ======================================
    // Future Layer Toggles
    // ======================================

    console.log("Dashboard Loaded");



    // ======================================
    // Analyze Button Loading Animation
    // ======================================

    const form = document.querySelector(".toolbar");

    if (form) {

        form.addEventListener("submit", function () {

            const btn = document.querySelector("button");

            btn.innerHTML = "Analyzing...";

            btn.disabled = true;

        });

    }

});