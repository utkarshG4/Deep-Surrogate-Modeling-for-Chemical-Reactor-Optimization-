"""Data loading and preprocessing utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import DATA_PROCESSED, INPUT_NAMES, OUTPUT_NAMES, SEED

def load_data():
    return pd.read_csv(DATA_PROCESSED / "reactor_dataset.csv")

def split_scale(df):
    X = df[INPUT_NAMES].values
    y = df[OUTPUT_NAMES].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    return (
        x_scaler.transform(X_train), x_scaler.transform(X_val), x_scaler.transform(X_test),
        y_scaler.transform(y_train), y_scaler.transform(y_val), y_scaler.transform(y_test),
        x_scaler, y_scaler
    )
