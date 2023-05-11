import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Carga el archivo CSV
data = pd.read_csv('/workspaces/Proyecto-COIL/sinavisos_recepcionados/aviso_recepcionados_limpio.csv', delimiter=',', encoding='utf-8')

# Aquí va el preprocesamiento y la preparación de los datos (no proporcionado en la pregunta)

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Define los rangos de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': np.arange(6, 22, 2),
    'learning_rate': [0.01, 0.1, 0.3, 0.5]
}

# Crea el modelo XGBClassifier
model = XGBClassifier(random_state=100)

# Realiza la búsqueda de hiperparámetros óptimos con GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Muestra los hiperparámetros óptimos encontrados
print("Mejores hiperparámetros encontrados: ", grid_search.best_params_)


# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Define los rangos de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': np.arange(6, 22, 2),
    'learning_rate': [0.01, 0.1, 0.3, 0.5]
}

# Crea el modelo XGBClassifier
model = XGBClassifier(random_state=100)

# Realiza la búsqueda de hiperparámetros óptimos con GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Muestra los hiperparámetros óptimos encontrados
print("Mejores hiperparámetros encontrados: ", grid_search.best_params_)


# Crea un nuevo modelo XGBClassifier con los hiperparámetros óptimos
best_model = grid_search.best_estimator_

# Entrena el modelo y realiza predicciones
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Calcula y muestra las métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, zero_division=0)

print("Accuracy: ", accuracy)
print("F1-score (weighted): ", f1)
print("Classification report: \n", class_report)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, train_sizes, cv=None, n_jobs=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Dibujar la Learning Curve
optimal_xgb = XGBClassifier(random_state=100, n_estimators=optimal_n_estimators, max_depth=optimal_max_depth, learning_rate=optimal_learning_rate)
train_sizes = np.linspace(start=1000, stop=X_train.shape[0], num=10, dtype=int)
plot_learning_curve(optimal_xgb, "Learning Curve for Optimal XGBClassifier", X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=-1)




