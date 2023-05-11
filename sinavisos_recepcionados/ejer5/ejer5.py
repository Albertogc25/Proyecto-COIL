import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier

# Cargar el archivo CSV
df = pd.read_csv('ruta/del/archivo.csv')

# Seleccionar características (columnas) y objetivo (target)
X = df.drop('CANAL_DE_ENTRADA_ID', axis=1)
y = df['CANAL_DE_ENTRADA_ID']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Rangos de hiperparámetros
n_estimators_range = [100, 200, 300, 400, 500]
max_depth_range = range(6, 21, 2)
learning_rate_options = [0.01, 0.1, 0.3, 0.5]

# Variables para almacenar los mejores valores encontrados
best_accuracy = 0
best_f1 = 0
best_model = None

# Búsqueda de los mejores hiperparámetros
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for learning_rate in learning_rate_options:
            model = XGBClassifier(random_state=100, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            if accuracy > best_accuracy and f1 > best_f1:
                best_accuracy = accuracy
                best_f1 = f1
                best_model = model

# Imprimir métricas del modelo óptimo
print("Optimal XGBClassifier Model:")
print("Accuracy:", best_accuracy)
print("F1-Score:", best_f1)
report = classification_report(y_test, y_pred, zero_division=0)
print("Classification Report:\n", report)

# Gráfico de Learning Curve para el modelo óptimo
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")
    plt.legend(loc="best")

# Graficar Learning Curve para el modelo óptimo
plot_learning_curve(best_model, "XGBClassifier Learning Curve", X, y, cv=5, n_jobs=-1)

# Mostrar todos los gráficos
plt.show()
