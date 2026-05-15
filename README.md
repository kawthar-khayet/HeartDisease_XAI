[README.md](https://github.com/user-attachments/files/27804261/README.md)
# 🫀 Heart Disease Prediction with XAI (Explainable AI)

## 📋 Project Overview

This project combines machine learning with **Explainable Artificial Intelligence (XAI)** to predict heart disease and provide interpretable insights into model predictions. The goal is to build robust predictive models while ensuring medical professionals and patients understand *why* predictions are made.

### ✨ Key Features
- 🤖 Multiple ML models (XGBoost, Random Forest, SVM, Decision Trees)
- 🔍 SHAP (SHapley Additive exPlanations) for global and local model interpretability
- 🎯 LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
- 📊 Interactive visualizations and dashboards
- 📈 Comprehensive performance metrics
- 🩺 Medical AI use case with explainability focus

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
