"""Tarea 4: preprocesamiento.

- Split estratificado train/test (80/20) con semilla fija para reproducibilidad.
- Escalado con StandardScaler ajustado SOLO sobre train (evita fuga de datos).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_data import NEWNAMES, TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [c for c in NEWNAMES if c != TARGET]


def split_scale(df: pd.DataFrame) -> tuple:
    """Devuelve X_train, X_test, y_train, y_test, scaler (escaladas)."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from clean_data import clean_mcar
    from load_data import load_data

    data = clean_mcar(load_data("Base_deficit_y_muertos_112017_conesposos.dta"))
    Xtr, Xte, ytr, yte, _ = split_scale(data)
    print(f"X_train {Xtr.shape} | X_test {Xte.shape}")
    print(f"diezmo target train: {np.bincount(ytr)} | test: {np.bincount(yte)}")