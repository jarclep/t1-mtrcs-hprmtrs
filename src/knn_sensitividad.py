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

    # Mismo rango de búsqueda de K que en el punto 5
    # Se usan números impares para evitar empates
    param_grid = {
        'n_neighbors': range(1, 51, 2)
    }

    # Cross Validation con GridSearch
    # Recall corresponde a la sensitividad
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='recall',
        cv=5,
        return_train_score=True
    )

    # Probar los diferentes valores de K
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

    # ---------------------------------------------------------
    # Primero vemos qué K obtiene el recall más alto
    # ---------------------------------------------------------

    k_max_recall = grid.best_params_['n_neighbors']
    indice_max = grid.best_index_

    recall_train_max = grid.cv_results_['mean_train_score'][indice_max]
    recall_val_max = grid.cv_results_['mean_test_score'][indice_max]

    print("\n--- Mayor sensitividad obtenida ---")
    print(f"K con mayor sensitividad: {k_max_recall}")
    print(f"Sensitividad entrenamiento: {recall_train_max:.4f}")
    print(f"Sensitividad validación:     {recall_val_max:.4f}")

    # ---------------------------------------------------------
    # Buscar un K que tenga buen recall pero sin una brecha
    # de sobreajuste mayor a 0.10
    # ---------------------------------------------------------

    resultados = grid.cv_results_

    mejor_k = None
    mejor_recall = -1
    mejor_indice = None

    for i, k in enumerate(resultados['param_n_neighbors']):

        recall_train = resultados['mean_train_score'][i]
        recall_val = resultados['mean_test_score'][i]

        brecha = recall_train - recall_val

        if brecha <= 0.10 and recall_val > mejor_recall:
            mejor_recall = recall_val
            mejor_k = int(k)
            mejor_indice = i

    # ---------------------------------------------------------
    # Mostrar el K seleccionado
    # ---------------------------------------------------------

    if mejor_k is not None:

        train_recall_cv = resultados['mean_train_score'][mejor_indice]
        val_recall_cv = resultados['mean_test_score'][mejor_indice]

        print("\n--- K seleccionado evitando sobreajuste ---")
        print(f"El K óptimo por sensitividad es: {mejor_k}")
        print(f"Sensitividad en entrenamiento: {train_recall_cv:.4f}")
        print(f"Sensitividad en validación:     {val_recall_cv:.4f}")

        brecha_final = train_recall_cv - val_recall_cv
        print(f"Brecha entrenamiento-validación: {brecha_final:.4f}")

        # Entrenar modelo final con el K seleccionado
        best_model = KNeighborsClassifier(
            n_neighbors=mejor_k
        )

        best_model.fit(X_train, y_train)

    else:

        # Por seguridad, si ningún K cumple el criterio,
        # usamos el mejor encontrado por GridSearch
        print("\nNingún K cumplió el criterio de brecha <= 0.10.")
        print("Se utilizará el K con mayor sensitividad.")

        mejor_k = k_max_recall
        best_model = grid.best_estimator_

    # ---------------------------------------------------------
    # Evaluación final en el conjunto de prueba
    # ---------------------------------------------------------

    y_pred = best_model.predict(X_test)

    test_recall = recall_score(y_test, y_pred)

    print("\n--- Evaluación final ---")
    print(f"K utilizado: {mejor_k}")
    print(
        f"Sensitividad final en el conjunto de prueba: "
        f"{test_recall:.4f}"
    )
