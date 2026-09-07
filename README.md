# Clasificador KNN — Fragilidad en adultos mayores (UNAM)

Método supervisado basado en distancias (KNN) sobre una muestra longitudinal
2012–2015 para predecir mortalidad en adultos mayores a partir de 35 déficits
(variables binarias de fragilidad).

## Datos

- `Base_deficit_y_muertos_112017_conesposos.dta` — muestra longitudinal
  (Stata). Incluye la variable objetivo `fallecido_15` (muerte entre 2012 y 2015).
- PDF de la actividad: `Actividad_clasificador_metricas_UNAM.pdf`

## Tareas

1. **Entorno** — repo git, dependencias, validación de lectura del `.dta`.
2. **Carga y selección** — 35 déficits + `fallecido_15`, renombrado legible.
3. **(a) Limpieza MCAR** — eliminar observaciones con pérdidas en cualquier variable.
4. **Preprocesamiento** — split estratificado (80/20, semilla fija) + `StandardScaler`.
5. **(b) KNN con precisión** — barrido de K, K óptimo sin sobreajuste.
6. **(c) KNN con sensitividad** — barrido de K por recall, comparativa con (b).
7. **Verificación final** — matrices de confusión, gráficas, reproducibilidad.

Los commits se mapean en `COMMITS.md`.