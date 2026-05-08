import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost entraîné")
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest entraîné")
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Decision Tree entraîné")
    return model

def train_svc(X_train_scaled, y_train):
    model = SVC(random_state=42, probability=True)
    model.fit(X_train_scaled, y_train)
    print("✅ SVC entraîné")
    return model

def evaluate_model(model, x_test, y_test, model_name='Modèle'):
    y_pred = model.predict(x_test)
    print(f"\n── {model_name} ──────────────────────")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision : {precision_score(y_test, y_pred):.2%}")
    print(f"Recall    : {recall_score(y_test, y_pred):.2%}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    return y_pred

def save_model(model, filename, folder='../models'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Modèle sauvegardé → {path}")

def load_model(filename, folder='../models'):
    path = os.path.join(folder, filename)
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé → {path}")
    return model
