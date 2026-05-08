import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(path='../data/heart.csv'):
    df = pd.read_csv(path)
    return df

def encode_features(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

def split_data(df, target='HeartDisease', test_size=0.2, random_state=43):
    X = df.drop(target, axis=1)
    Y = df[target]
    X_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, x_test, y_train, y_test

def scale_data(X_train, x_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled  = scaler.transform(x_test)
    return X_train_scaled, x_test_scaled, scaler

def save_processed_data(X_train, x_test, y_train, y_test):
    os.makedirs('../data/processed', exist_ok=True)
    X_train.to_csv('../data/processed/X_train_ready.csv', index=False)
    x_test.to_csv('../data/processed/X_test_ready.csv',   index=False)
    y_train.to_csv('../data/processed/y_train.csv',        index=False)
    y_test.to_csv('../data/processed/y_test.csv',          index=False)
    print("✅ Données sauvegardées dans data/processed/")
