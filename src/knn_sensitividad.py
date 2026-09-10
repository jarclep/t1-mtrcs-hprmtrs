"""Tarea 6: KNN con sensitividad."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

from load_data import load_data
from clean_data import clean_mcar
from preprocess import split_scale


def best_knn_sensitividad(X_train, y_train):

    knn = KNeighborsClassifier()

    parametros = {
        'n_neighbors': range(1, 51, 2)
    }

    grid = GridSearchCV(
        knn,
        parametros,
        scoring='recall',
        cv=5,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    return grid


if __name__ == "__main__":

    print("Cargando datos")

    data = load_data(
        "Base_deficit_y_muertos_112017_conesposos.dta"
    )

    data = clean_mcar(data)

    X_train, X_test, y_train, y_test, scaler = split_scale(data)

    print("Probando valores de K")

    grid = best_knn_sensitividad(X_train, y_train)

    resultados = grid.cv_results_

    mejor_k = None
    mejor_recall = -1
    mejor_indice = None

    for i, k in enumerate(resultados['param_n_neighbors']):

        recall_train = resultados['mean_train_score'][i]
        recall_val = resultados['mean_test_score'][i]

        diferencia = recall_train - recall_val

        if diferencia <= 0.10 and recall_val > mejor_recall:
            mejor_recall = recall_val
            mejor_k = int(k)
            mejor_indice = i


    if mejor_k is not None:

        recall_train = resultados['mean_train_score'][mejor_indice]
        recall_val = resultados['mean_test_score'][mejor_indice]

        print(f"K óptimo: {mejor_k}")
        print(f"Sensitividad entrenamiento: {recall_train:.4f}")
        print(f"Sensitividad validación: {recall_val:.4f}")

        modelo = KNeighborsClassifier(
            n_neighbors=mejor_k
        )

        modelo.fit(X_train, y_train)

    else:

        mejor_k = grid.best_params_['n_neighbors']
        modelo = grid.best_estimator_

        print(f"K con mayor sensitividad: {mejor_k}")


    y_pred = modelo.predict(X_test)

    sensitividad_test = recall_score(
        y_test,
        y_pred
    )

    print(f"Sensitividad en prueba: {sensitividad_test:.4f}")
