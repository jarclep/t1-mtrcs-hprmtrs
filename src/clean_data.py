"""Tarea 3 (a): limpieza bajo esquema MCAR.

Se eliminan todas las observaciones con pérdidas en cualquiera de las variables.
"""
import pandas as pd


def clean_mcar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con any valor nulo. Reporta el texto del antes/después."""
    n_before = len(df)
    clean = df.dropna().copy()
    n_after = len(clean)
    print(f"antes del dropna: {n_before} obs | después: {n_after} "
          f"| eliminadas: {n_before - n_after}")
    return clean


if __name__ == "__main__":
    from load_data import load_data

    data = load_data("Base_deficit_y_muertos_112017_conesposos.dta")
    data = clean_mcar(data)
    print(f"distribución target:\n{data['Dead'].value_counts().to_string()}")
    print(f"nulos restantes: {int(data.isna().sum().sum())}")