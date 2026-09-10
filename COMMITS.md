# Mapeo de commits por tarea

Este archivo lista, por tarea, los commits que la cubren. Se actualiza de forma
incremental tras cada commit.

## Tarea 1 — Entorno

| # | Hash     | Fecha       | Mensaje / Slice                                                |
|---|----------|-------------|----------------------------------------------------------------|
| 1 | `f6c07c5` | 2026-09-07  | init: repo git + estructura basica (.gitignore, README)        |
| 2 | `a302f4d` | 2026-09-07  | deps: venv con pandas/numpy/scikit-learn, validacion lectura .dta (14707x82) |

## Tarea 2 — Carga y selección

| # | Hash     | Fecha      | Mensaje / Slice                                      |
|---|----------|------------|------------------------------------------------------|
| 1 | `305138d` | 2026-09-07 | src/load_data.py: 36 vars de myvars2, renombrado, target Dead |

## Tarea 3 — (a) Limpieza MCAR

| # | Hash     | Fecha      | Mensaje / Slice                                             |
|---|----------|------------|-------------------------------------------------------------|
| 1 | `a2ee1d3` | 2026-09-07 | src/clean_data.py: dropna MCAR (14707 -> 10983 obs)         |

## Tarea 4 — Preprocesamiento

| # | Hash     | Fecha      | Mensaje / Slice                                             |
|---|----------|------------|-------------------------------------------------------------|
| 1 | `f772b27` | 2026-09-07 | src/preprocess.py: split estratificado 80/20 (seed 42) + StandardScaler |

## Tarea 5 — (b) KNN con precisión

| # | Hash | Fecha | Mensaje / Slice |
|---|------|-------|-----------------|
| 1| 'ba317be' |2026-09-09 | Knn con metrica precision, K=29|

## Tarea 6 — (c) KNN con sensitividad

| # | Hash | Fecha | Mensaje / Slice |
|---|------|-------|-----------------|
| 1 | `815c16a` | 2026-09-09 | KNN con métrica sensitividad, K=5 |

## Tarea 7 — Verificación final

| # | Hash | Fecha | Mensaje / Slice |
|---|------|-------|-----------------|
| 1 | `77af910` | 2026-09-09 | Verificación final de modelos KNN |
