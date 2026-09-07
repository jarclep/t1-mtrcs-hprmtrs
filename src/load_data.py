"""Tarea 2: carga, selección de variables y renombrado legible.

Reproduce la correspondencia del PDF (miembros 'myvars2' y 'colnames'):
- 36 columnas seleccionadas del .dta (35 déficits + fallecido_15).
- Renombrado a nombres indicativos; 'fallecido_15' pasa a ser 'Dead' (target).
"""
from pathlib import Path

import pandas as pd

MYVARS2 = [
    "h13_12_d_2", "h5_12_d_2", "h15d_12_d", "h17d_12_d",
    "h16d_12_d", "h19d_12_d", "h7_12_d_2", "h11_12_d_2", "h27c_12_d_2",
    "h26c_12_d_2", "h28c_12_d_2", "h29c_12_d_2", "c64_12_d", "c1_12_d",
    "c2a_12_d", "c73_12_d", "c49_8_12_d", "h3_12_d_2", "c49_2_12_d",
    "c49_1_12_d", "c49_4_12_d", "c49_5_12_d", "c49_9_12_d", "c4_12_d",
    "c22a_12_d", "c25b_12_d", "c26_12_d", "c12_12_menos10_d",
    "c6_12_d", "c32_12_d", "c19_12_d", "e1b_12_d", "c69a_12_d",
    "fallecido_15", "c70_12_d", "c50b_12_d",
]

NEWNAMES = [
    "Dress", "In_out_chair", "Walk", "Eat", "Groom",
    "Toilet", "Stairs", "Lift", "Shop", "Meals", "Meds", "Finance",
    "Lost_weight", "Health", "H_change", "Bed", "Tired", "Walk_out",
    "Effort", "Depressed", "No_Happy", "Lone", "No_Energy", "High_BP",
    "Heart_attack", "CHF", "Stroke", "Cancer", "Diabetes", "Arthritis",
    "CLD", "Memory", "Grip", "Dead", "Anorexia", "Exercise",
]

TARGET = "Dead"


def load_data(path: str | Path) -> pd.DataFrame:
    """Lee el .dta, selecciona las 36 variables y las renombra."""
    raw = pd.read_stata(path)
    missing = [c for c in MYVARS2 if c not in raw.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el .dta: {missing}")
    df = raw[MYVARS2].copy()
    df.columns = NEWNAMES
    return df


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "Base_deficit_y_muertos_112017_conesposos.dta"
    data = load_data(path)
    print(f"shape: {data.shape}")
    print(f"target '{TARGET}':\n{data[TARGET].value_counts().to_string()}")
    print(f"valores nulos por columna: {int(data.isna().sum().sum())}")