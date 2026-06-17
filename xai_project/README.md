#  Heart Disease Prediction with XAI (Explainable AI)

##  Project Overview

This project combines machine learning with **Explainable Artificial Intelligence (XAI)** to predict heart disease and provide interpretable insights into model predictions. The goal is to build robust predictive models while ensuring medical professionals and patients understand *why* predictions are made.

###  Key Features
-  Multiple ML models (XGBoost, Random Forest, SVM, Decision Trees)
-  SHAP (SHapley Additive exPlanations) for global and local model interpretability
-  LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
-  Interactive visualizations and dashboards
-  Comprehensive performance metrics
-  Medical AI use case with explainability focus

---

# 📁 Project Structure

```bash
xai_project/
│
├── data/                         # Dataset storage
│   ├── heart.csv                 # Raw heart disease dataset
│   └── processed/                # Preprocessed data
│
├── src/                          # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── preprocessing.py          # Data loading & preprocessing
│   ├── models.py                 # Model training & evaluation
│   └── explainability.py         # SHAP & LIME explanations
│
├── notebooks/                    # Jupyter notebooks
├── models/                       # Saved trained models
│
├── outputs/
│   └── figures/                  # Generated plots & explanations
│
├── dashboard/                    # Streamlit / Plotly dashboards
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

```
---
##  Dataset
 
The project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (`heart.csv`).
 
| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |
| `target` | 1 = Heart disease, 0 = No disease |
 
---
 
##  Setup
 
### Prerequisites
 
- Python 3.9+
- pip
### Installation
 
```bash
git clone https://github.com/YOUR_USERNAME/xai-heart-disease.git
cd xai-heart-disease
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
---
 
##  Usage
 
### 1. Preprocessing
 
```bash
python src/preprocessing.py
```
 
Loads `data/heart.csv`, handles missing values, encodes features, and saves processed data to `data/processed/`.
 
### 2. Train Models
 
```bash
python src/models.py
```
 
Trains XGBoost, Random Forest, SVM, and Decision Tree classifiers. Saves models to `models/` and prints evaluation metrics.
 
### 3. Generate Explanations
 
```bash
python src/explainability.py
```
 
Runs SHAP (global feature importance, beeswarm plots, waterfall charts) and LIME (per-patient local explanations). Saves visuals to `outputs/figures/`.
 
### 4. Launch Dashboard
 
```bash
streamlit run dashboard/app.py
```
 
Opens an interactive dashboard to explore predictions and explanations patient by patient.
 
---
 

 
 
---
 
##  Technologies
 
| Library | Role |
|---|---|
| [scikit-learn](https://scikit-learn.org/) | ML models & evaluation |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting classifier |
| [SHAP](https://shap.readthedocs.io/) | Global & local model explanations |
| [LIME](https://github.com/marcotcr/lime) | Per-prediction local explanations |
| [Streamlit](https://streamlit.io/) | Interactive dashboard |
| [Plotly](https://plotly.com/python/) | Interactive visualizations |
| [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) | Data processing |
 
---
 
##  Medical Context
 
Explainability is critical in healthcare AI. This project ensures that every prediction comes with a human-readable justification — enabling clinicians to validate, question, or override model decisions rather than treating the model as a black box.
