"""Tarea 6, (c) en el PDF: KNN con métrica sensitividad.

Se repite el análisis utilizando la sensitividad como métrica
para determinar la eficiencia del clasificador y el valor óptimo de K.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

from load_data import load_data
from clean_data import clean_mcar
from preprocess import split_scale


def best_knn_sensitividad(X_train, y_train):

    knn = KNeighborsClassifier()

    # Mismo rango de K que en el punto 5
    # Se usan impares para evitar empates entre vecinos
    param_grid = {'n_neighbors': range(1, 51, 2)}

    # Cross Validation con GridSearch
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='recall',
        cv=5,
        return_train_score=True
    )

    # Probando diferentes valores de K
    grid_search.fit(X_train, y_train)

    return grid_search


if __name__ == "__main__":

    print("Cargando y procesando datos")

    data = clean_mcar(
        load_data("Base_deficit_y_muertos_112017_conesposos.dta")
    )

    X_train, X_test, y_train, y_test, scaler = split_scale(data)

    print("Barriendo K con Cross Validation")

    grid = best_knn_sensitividad(X_train, y_train)

    best_k = grid.best_params_['n_neighbors']

    print(f"El K óptimo por sensitividad es: {best_k}")

    best_index = grid.best_index_

    train_recall_cv = grid.cv_results_['mean_train_score'][best_index]
    val_recall_cv = grid.cv_results_['mean_test_score'][best_index]

    print(f"Sensitividad en entrenamiento: {train_recall_cv:.4f}")
    print(f"Sensitividad en validación:    {val_recall_cv:.4f}")

    if (train_recall_cv - val_recall_cv) > 0.10:
        print("La brecha es grande, podría haber cierto sobreajuste.")
    else:
        print("La brecha es pequeña. No hay sobreajuste grande.")

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    test_recall = recall_score(y_test, y_pred)

    print(
        f"\nSensitividad final en el conjunto de prueba: "
        f"{test_recall:.4f}"
    )
