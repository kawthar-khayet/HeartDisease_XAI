import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Configuration ─────────────────────────────
st.set_page_config(
    page_title="CardioCheck XAI",
    page_icon="🫀",
    layout="wide"
)

# ── Chargement modèle et données ──────────────
@st.cache_resource
def load_resources():
    with open('../models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X_train = pd.read_csv('../data/processed/X_train_ready.csv')
    x_test  = pd.read_csv('../data/processed/X_test_ready.csv')
    y_test  = pd.read_csv('../data/processed/y_test.csv').squeeze()
    return model, X_train, x_test, y_test

model, X_train, x_test, y_test = load_resources()

# ── Header ────────────────────────────────────
st.title("🫀 CardioCheck XAI")
st.markdown("**Prédiction et explication des maladies cardiaques**")
st.divider()

# ── Sidebar — saisie patient ──────────────────
st.sidebar.header("📋 Données du patient")

age            = st.sidebar.slider("Âge",                    20, 100, 55)
sex            = st.sidebar.selectbox("Sexe",                [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
chest_pain     = st.sidebar.selectbox("Type douleur (0=ASY, 1=ATA, 2=NAP, 3=TA)", [0, 1, 2, 3])
resting_bp     = st.sidebar.slider("Tension au repos",       80, 200, 120)
cholesterol    = st.sidebar.slider("Cholestérol",            0, 600, 200)
fasting_bs     = st.sidebar.selectbox("Glycémie à jeun > 120", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
resting_ecg    = st.sidebar.selectbox("ECG au repos (0/1/2)", [0, 1, 2])
max_hr         = st.sidebar.slider("Fréquence cardiaque max", 60, 220, 150)
exercise_angina= st.sidebar.selectbox("Angine à l'effort",  [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
oldpeak        = st.sidebar.slider("Oldpeak",                0.0, 6.0, 0.0, step=0.1)
st_slope       = st.sidebar.selectbox("ST Slope (0=Down, 1=Flat, 2=Up)", [0, 1, 2])

# ── Données patient ───────────────────────────
patient_data = pd.DataFrame([{
    'Sex':            sex,
    'ChestPainType':  chest_pain,
    'RestingBP':      resting_bp,
    'Cholesterol':    cholesterol,
    'FastingBS':      fasting_bs,
    'RestingECG':     resting_ecg,
    'MaxHR':          max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak':        oldpeak,
    'ST_Slope':       st_slope,
    'Age':            age,
}])[X_train.columns]  # respecter l'ordre des colonnes

# ── Prédiction ────────────────────────────────
st.header("🔍 Résultat de la prédiction")

prediction  = model.predict(patient_data)[0]
probability = model.predict_proba(patient_data)[0][1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Prédiction", "🔴 Maladie" if prediction == 1 else "🟢 Sain")
with col2:
    st.metric("Probabilité maladie", f"{probability:.1%}")
with col3:
    st.metric("Confiance", f"{max(probability, 1-probability):.1%}")

st.progress(float(probability))
st.divider()

# ── SHAP ──────────────────────────────────────
st.header("📊 Explication SHAP")

explainer_shap = shap.TreeExplainer(model)
shap_values    = explainer_shap.shap_values(patient_data)

fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer_shap.expected_value,
    data=patient_data.iloc[0],
    feature_names=patient_data.columns.tolist()
), show=False)
st.pyplot(fig)
plt.close()

st.divider()

# ── LIME ──────────────────────────────────────
st.header("🧪 Explication LIME")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['Pas de maladie', 'Maladie cardiaque'],
    mode='classification'
)

explanation = explainer_lime.explain_instance(
    data_row=np.array(patient_data)[0],
    predict_fn=model.predict_proba,
    num_features=10
)

fig2 = explanation.as_pyplot_figure()
st.pyplot(fig2)
plt.close()

st.divider()

# ── Footer ────────────────────────────────────
st.caption("CardioCheck XAI — Projet d'explicabilité des modèles de ML ")
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_resources():
    with open(os.path.join(BASE_DIR, 'models', 'xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_train_ready.csv'))
    x_test  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_test_ready.csv'))
    y_test  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'y_test.csv')).squeeze()
    return model, X_train, x_test, y_test
