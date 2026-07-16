# 🌍 AI-Based Multi-Layer Terrain and Land Cover Change Detection System

An AI-assisted terrain monitoring system that performs multi-temporal satellite image analysis to detect and visualize changes in land cover. The project provides layer-wise terrain analysis through an interactive Flask-based web application.

---

## 🚀 Features

- Vegetation Change Detection
- Water Body Detection
- Wetland Detection
- Marshy Area Estimation
- Boggy Area Estimation
- Terrain Analysis
- Road Accessibility Analysis
- Change Heatmap Generation
- Combined Terrain Overlay
- AI-Based Terrain Summary
- Layer-wise Visualization
- Before & After Image Comparison

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|-----------|
| Python | Backend Logic |
| Flask | Web Application Framework |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Matplotlib | Heatmap Visualization |
| HTML, CSS & JavaScript | Frontend Development |
| Satellite Imagery | Terrain Analysis |

---

## 📂 Project Structure

```text
Terrain-Change-Detection-System/

│
├── app.py
├── requirements.txt
│
├── models/
│   ├── vegetation.py
│   ├── water.py
│   ├── wetlands.py
│   ├── marshy.py
│   ├── boggy.py
│   ├── terrain_analytics.py
│   └── change_detector.py
│
├── static/
│
├── templates/
│
├── outputs/
│
└── sample_images/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-link>
```

Move into the project directory:

```bash
cd Terrain-Change-Detection-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Flask application using:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🔍 Workflow

```text
Satellite Images
        ↓
Vegetation Detection
        ↓
Water Body Detection
        ↓
Wetland Detection
        ↓
Marshy & Boggy Area Estimation
        ↓
Terrain Analysis
        ↓
Road Accessibility Analysis
        ↓
Change Detection
        ↓
Heatmap Generation
        ↓
Layer-wise Visualization
        ↓
AI Summary Generation
```

---

## 📊 Output

The system generates:

- Vegetation Overlay
- Water Overlay
- Wetland Overlay
- Combined Overlay
- Change Heatmap
- Terrain Analysis
- Road Accessibility Analysis
- AI Summary
- Metadata Statistics
- Before & After Image Comparison

---

## 📸 Sample Results

You can add screenshots of:

- Home Page
- Vegetation Detection
- Water Detection
- Wetland Detection
- Heatmap Visualization
- Terrain Analysis Dashboard

---

## ⚠️ Limitations

- Water body detection may vary depending on satellite image quality and terrain conditions.
- Seasonal variations in satellite imagery can affect terrain classification.
- The current implementation uses image processing techniques and can be further enhanced using semantic segmentation models.

---

## 🔮 Future Scope

- Integration of Semantic Segmentation Models.
- Elevation Change Detection.
- Advanced Road Detection Techniques.
- Real-Time Satellite Image Analysis.
- Multi-Class Land Cover Classification.

---

## 👩‍💻 Author

**Nishtha Priya**

B.Tech - Computer Science Engineering (AI & ML)  
Bennett University

---

## 📜 License

This project has been developed as part of the Bharat Electronics Limited (BEL) Internship Project.