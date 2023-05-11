import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Leer el archivo CSV (asumiendo que está en el mismo directorio que este archivo)
df = pd.read_csv('/workspaces/Proyecto-COIL/sinavisos_recepcionados/aviso_recepcionados_limpio.csv', sep=',')

# Preparar los datos
X = df.drop(['TIPO_INCIDENCIA'], axis=1)
y = df['TIPO_INCIDENCIA']

# Dividir el dataset en training y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Entrenar el modelo de regresión logística
logistic_regression = LogisticRegression(max_iter=1000, random_state=100)
logistic_regression.fit(X_train, y_train)

# Hacer predicciones en el conjunto de test
y_pred = logistic_regression.predict(X_test)

# Calcular métricas para evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar los resultados
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Classification report:\n", report)
print("Confusion matrix:\n", conf_matrix)
