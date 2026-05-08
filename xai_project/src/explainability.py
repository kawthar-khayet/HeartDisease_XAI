import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('../outputs/figures', exist_ok=True)

# ── SHAP ─────────────────────────────────────────────────────

def shap_summary_bar(model, x_test):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('../outputs/figures/shap_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP bar plot sauvegardé")
    return explainer, shap_values

def shap_summary_beeswarm(shap_values, x_test):
    shap.summary_plot(shap_values, x_test, show=False)
    plt.tight_layout()
    plt.savefig('../outputs/figures/shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP beeswarm plot sauvegardé")

def shap_waterfall(explainer, shap_values, x_test, patient_index=0):
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[patient_index],
        base_values=explainer.expected_value,
        data=x_test.iloc[patient_index],
        feature_names=x_test.columns.tolist()
    ), show=False)
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/shap_patient_{patient_index}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP waterfall patient {patient_index} sauvegardé")

# ── LIME ─────────────────────────────────────────────────────

def lime_explanation(model, X_train, x_test, y_test, patient_index=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=['Pas de maladie', 'Maladie cardiaque'],
        mode='classification'
    )
    patient     = np.array(x_test)[patient_index]
    explanation = explainer.explain_instance(
        data_row=patient,
        predict_fn=model.predict_proba,
        num_features=10
    )
    print(f"Prédiction réelle   : {y_test.iloc[patient_index]}")
    print(f"Prédiction modèle   : {model.predict([patient])[0]}")
    print(f"Probabilité maladie : {model.predict_proba([patient])[0][1]:.2%}")

    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/lime_patient_{patient_index}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ LIME patient {patient_index} sauvegardé")
    return explanation
