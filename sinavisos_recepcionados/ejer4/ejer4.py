import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Leer el archivo CSV
df = pd.read_csv('/workspaces/Proyecto-COIL/sinavisos_recepcionados/aviso_recepcionados_limpio.csv', sep=',', dtype=str)

# Realizar preprocesamiento y dividir en conjuntos de entrenamiento y prueba
# ... Aquí va el código de preprocesamiento ...
X = df.drop('CANAL_DE_ENTRADA_ID', axis=1)  # Reemplaza 'target_column_name' con el nombre de tu columna objetivo
y = df['CANAL_DE_ENTRADA_ID']  # Reemplaza 'target_column_name' con el nombre de tu columna objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 1. Entrenar un modelo tipo Random Forest Classifier y calcular métricas
rf = RandomForestClassifier(random_state=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, zero_division=0)

print("Random Forest Metrics:")
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Classification Report:", report)

# 2. Calcular la mediana de la profundidad de los árboles en el bosque
tree_depths = [tree.tree_.max_depth for tree in rf.estimators_]
median_depth = np.median(tree_depths)
print("Mediana de la profundidad de los árboles:", median_depth)

# 3. Curvas de complejidad del modelo Random Forest y nuevo clasificador con valores óptimos
param_grid = {
    'n_estimators': np.array([200, 250, 300, 350, 400]),
    'max_depth': np.arange(20, 42, 2),
    'max_features': ['auto', 'log2', None]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=100), param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

y_pred_optimized = best_rf.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
f1_optimized = f1_score(y_test, y_pred_optimized, average='weighted')
report_optimized = classification_report(y_test, y_pred_optimized, zero_division=0)

print("Optimized Random Forest Metrics:")
print("Accuracy:", accuracy_optimized)
print("F1-score:", f1_optimized)
print("Classification Report:", report_optimized)
print("OOB Score:", best_rf.oob_score_)

# 4. Gráfica del Learning Curve para Random Forest
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt

# 5. Entrenar un modelo tipo Decision Tree Classifier y calcular métricas
dt = DecisionTreeClassifier(random_state=100)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
report_dt = classification_report(y_test, y_pred_dt, zero_division=0)

print("Decision Tree Metrics:")
print("Accuracy:", accuracy_dt)
print("F1-score:", f1_dt)
print("Classification Report:", report_dt)

# 6. Curvas de complejidad del modelo Decision Tree y nuevo clasificador con valores óptimos
param_grid_dt = {
    'max_depth': np.arange(2, 31),
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid_dt, scoring='accuracy', cv=5, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

best_dt = grid_search_dt.best_estimator_
best_dt.fit(X_train, y_train)

y_pred_dt_optimized = best_dt.predict(X_test)
accuracy_dt_optimized = accuracy_score(y_test, y_pred_dt_optimized)
f1_dt_optimized = f1_score(y_test, y_pred_dt_optimized, average='weighted')
report_dt_optimized = classification_report(y_test, y_pred_dt_optimized, zero_division=0)

print("Optimized Decision Tree Metrics:")
print("Accuracy:", accuracy_dt_optimized)
print("F1-score:", f1_dt_optimized)
print("Classification Report:", report_dt_optimized)

# 7. Gráfica del Learning Curve para Decision Tree
title_dt = "Learning Curves (Decision Tree)"
cv_dt = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

plt_obj_dt = plot_learning_curve(best_dt, title_dt, X, y, cv=cv_dt, n_jobs=4)

# Mostrar la gráfica del Learning Curve para Decision Tree
plt_obj_dt.show()

