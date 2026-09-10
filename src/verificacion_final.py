"""Tarea 7: verificación final de los modelos KNN."""

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score
)

from load_data import load_data
from clean_data import clean_mcar
from preprocess import split_scale


# Cargar y preparar los datos
data = load_data(
    "Base_deficit_y_muertos_112017_conesposos.dta"
)

data = clean_mcar(data)

X_train, X_test, y_train, y_test, scaler = split_scale(data)


# Modelo del punto 5: optimizado por precisión
modelo_precision = KNeighborsClassifier(
    n_neighbors=29
)

modelo_precision.fit(X_train, y_train)

pred_precision = modelo_precision.predict(X_test)


# Modelo del punto 6: optimizado por sensitividad
modelo_sensitividad = KNeighborsClassifier(
    n_neighbors=5
)

modelo_sensitividad.fit(X_train, y_train)

pred_sensitividad = modelo_sensitividad.predict(X_test)


# Métricas
precision_k29 = precision_score(
    y_test,
    pred_precision,
    zero_division=0
)

sensitividad_k29 = recall_score(
    y_test,
    pred_precision
)

precision_k5 = precision_score(
    y_test,
    pred_sensitividad,
    zero_division=0
)

sensitividad_k5 = recall_score(
    y_test,
    pred_sensitividad
)


print("MODELO K=29")
print(f"Precisión: {precision_k29:.4f}")
print(f"Sensitividad: {sensitividad_k29:.4f}")

print("\nMODELO K=5")
print(f"Precisión: {precision_k5:.4f}")
print(f"Sensitividad: {sensitividad_k5:.4f}")


# Matriz de confusión para K=29
matriz_29 = confusion_matrix(
    y_test,
    pred_precision
)

print("\nMatriz de confusión K=29")
print(matriz_29)

ConfusionMatrixDisplay(
    confusion_matrix=matriz_29
).plot()

plt.title("Matriz de confusión - K=29")
plt.show()


# Matriz de confusión para K=5
matriz_5 = confusion_matrix(
    y_test,
    pred_sensitividad
)

print("\nMatriz de confusión K=5")
print(matriz_5)

ConfusionMatrixDisplay(
    confusion_matrix=matriz_5
).plot()

plt.title("Matriz de confusión - K=5")
plt.show()


# Comparación de métricas
modelos = ["K=29", "K=5"]

precision = [
    precision_k29,
    precision_k5
]

sensitividad = [
    sensitividad_k29,
    sensitividad_k5
]

x = range(len(modelos))

plt.bar(
    [i - 0.2 for i in x],
    precision,
    width=0.4,
    label="Precisión"
)

plt.bar(
    [i + 0.2 for i in x],
    sensitividad,
    width=0.4,
    label="Sensitividad"
)

plt.xticks(x, modelos)
plt.ylabel("Valor de la métrica")
plt.title("Comparación de modelos KNN")
plt.legend()
plt.show()
