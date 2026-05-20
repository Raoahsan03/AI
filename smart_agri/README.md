# Smart Agriculture Decision Support System

> **Course:** Artificial Intelligence — OEL [CLO-2]  
> **Bahria University, Islamabad Campus — Dept. of Software Engineering**

A production-grade, multi-model AI pipeline that integrates **Decision Tree Classification**, **KMeans Clustering**, and **Linear Regression** into a unified Tkinter GUI for precision agriculture.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Presentation Layer                      │
│              src/gui.py  (Tkinter + Matplotlib)          │
└────────────────────────┬─────────────────────────────────┘
                         │ loads serialized models
┌────────────────────────▼─────────────────────────────────┐
│                    Model Layer                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ Decision Tree│ │    KMeans    │ │ Linear Regression│ │
│  │ Crop Rec.    │ │ Soil Zones   │ │ Yield Prediction │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
│              src/models.py + models/*.pkl                │
└────────────────────────┬─────────────────────────────────┘
                         │ preprocessed arrays
┌────────────────────────▼─────────────────────────────────┐
│                    Data Layer                            │
│   src/preprocessing.py  →  data/crop_data.csv           │
│   (22 crops · 2200 samples · 7 features + yield)        │
└──────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/<your-username>/smart-agri-dss.git
cd smart-agri-dss
pip install -r requirements.txt
```

Requires **Python 3.9+** and standard pip.

---

## Usage

**Step 1 — Train models** (generates `models/` and `results/`):
```bash
python train_models.py
```

**Step 2 — Launch the GUI**:
```bash
python -m src.gui
```

---

## Repository Structure

```
smart_agri/
├── data/                    # Dataset and metadata
│   └── crop_data.csv
├── src/                     # Modular source code
│   ├── preprocessing.py     # Data generation, cleaning, scaling
│   ├── models.py            # Model training and serialization
│   ├── gui.py               # Tkinter graphical interface
│   └── utils.py             # Constants and helper functions
├── models/                  # Serialized model artifacts (.pkl)
├── results/                 # Evaluation plots + metrics JSON
├── train_models.py          # CLI training pipeline
├── requirements.txt         # Dependency manifest
├── LICENSE                  # MIT License
└── README.md
```

---

## Algorithmic Rationale

| Model | Task | Rationale |
|---|---|---|
| **Decision Tree** | Crop recommendation (22 classes) | Interpretable, handles mixed feature scales, importance scores expose domain logic |
| **KMeans Clustering** | Soil zone segmentation | Unsupervised partitioning into homogeneous nutrient zones; silhouette score guides k selection |
| **Linear Regression** | Crop yield prediction (kg/ha) | Establishes a direct, explainable mapping from soil/climate inputs to quantitative output |

---

## Dataset

**Source:** Synthetic dataset derived from published agronomic parameter ranges (mimics the Kaggle *Crop Recommendation Dataset*).

**Features (7):**

| Feature | Unit | Description |
|---|---|---|
| N | mg/kg | Soil Nitrogen content |
| P | mg/kg | Soil Phosphorous content |
| K | mg/kg | Soil Potassium content |
| temperature | °C | Average ambient temperature |
| humidity | % | Relative humidity |
| ph | — | Soil pH (0–14) |
| rainfall | mm | Annual rainfall |

**Targets:**
- `label` — crop type (22 classes, classification)
- `yield` — crop yield in kg/ha (regression)

**Preprocessing pipeline:**
1. Median imputation for missing values
2. 1st–99th percentile capping for outlier treatment
3. StandardScaler normalisation
4. Stratified 80/20 train/test split

---

## Performance Summary

| Model | Metric | Value |
|---|---|---|
| Decision Tree | Accuracy | ~0.94 |
| Decision Tree | Weighted F1 | ~0.94 |
| KMeans | Silhouette Score | ~0.35 |
| Linear Regression | RMSE | ~2 800 kg/ha |
| Linear Regression | R² | ~0.76 |

*(Exact values printed by `train_models.py` and stored in `results/metrics_summary.json`)*

---

## Future Work

1. **IoT Sensor Integration:** Replace manual parameter entry with real-time MQTT streams from field-deployed soil and microclimate sensors, enabling continuous inference and alert-based crop management.

2. **Ensemble Deep Learning:** Replace the single Decision Tree classifier with a Random Forest or LightGBM ensemble, and augment yield prediction using an LSTM network trained on multi-year time-series weather and satellite NDVI imagery for spatiotemporal yield forecasting.

---

## License

MIT — see [LICENSE](LICENSE).
