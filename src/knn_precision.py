"""Tarea 5, (b) en el PDF: KNN con métrica precisión.

b  Dado que el número de individuos que mueren es bajo respecto a los que no, 
utiliza primero la precisión (recordar que este concepto es diferente a la exactitud o accuracy)
como métrica para determinar la eficiencia del clasificador. 
Recuerda variar el valor de K hasta un valor óptimo, o sea que no haya sobreajuste, 
pero que a la vez se obtengan buenos valores en la métrica de interés.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score


from load_data import load_data
from clean_data import clean_mcar
from preprocess import split_scale

def best_knn_precision(X_train, y_train):

    knn = KNeighborsClassifier()

    # diccionario con el rango de busqueda de K
    # impares para que no se empaten los vecinos 
    param_grid = {'n_neighbors': range(1, 51, 2)}

    #Cross validation con GridSearch
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        # la metrica
        scoring='precision',
        cv=5,                 
        return_train_score=True
        )

    # probando Ks 
    grid_search.fit(X_train, y_train)

    return grid_search

if __name__ == "__main__":

    print("Cargando y procesando datos")
    data = clean_mcar(load_data("Base_deficit_y_muertos_112017_conesposos.dta"))
    X_train, X_test, y_train, y_test, scaler = split_scale(data)

    print("Barriendo de K con Cross Validation")
    grid = best_knn_precision(X_train, y_train)

    best_k = grid.best_params_['n_neighbors']
    print(f"El K +optimo es: {best_k}")

    best_index = grid.best_index_
    

    train_precision_cv = grid.cv_results_['mean_train_score'][best_index]
    val_precision_cv = grid.cv_results_['mean_test_score'][best_index]
    
    print(f"Precisión en entrenamiento: {train_precision_cv:.4f}")
    print(f"Precisión en validación:    {val_precision_cv:.4f}")
    
    if (train_precision_cv - val_precision_cv) > 0.10:
        print("La brecha es grande, podría haber cierto sobreajuste.")
    else:
        print("La brecha es pequeña. No hay sobreajuste grande")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    test_precision = precision_score(y_test, y_pred)
    
    print(f"\nPrecisión final en el conjunto de Prueba: {test_precision:.4f}")
