import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Leer el archivo CSV
df = pd.read_csv('/workspaces/Proyecto-COIL/sinavisos_recepcionados/aviso_recepcionados_limpio.csv', sep=',', low_memory=False)

# Convertir las columnas con tipos mixtos a cadenas
df['TIPO_INCIDENCIA'] = df['TIPO_INCIDENCIA'].astype(str)
df['CATEGORIA_NIVEL1'] = df['CATEGORIA_NIVEL1'].astype(str)

# 1. Generar gráfica para visualizar distribución de las variables del dataset
numerical_features = df.select_dtypes(include=[np.number]).columns
n_features = len(numerical_features)

# Dividir las variables numéricas en grupos de 20
grouped_features = [numerical_features[i:i + 20] for i in range(0, n_features, 20)]

# Crear un histograma para cada grupo de 20 variables y guardar en archivos PNG
for idx, group in enumerate(grouped_features):
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(group):
        plt.subplot(5, 4, i + 1)
        plt.hist(df[feature], bins=50)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(f'histogram_group_{idx + 1}.png')
    plt.close()

# Analizar si hay necesidad de normalizar los datos
print(df.describe())

# 2. Normalizar todas las variables del dataset
# Llevar las variables de entrada a una escala de 0 a 1
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Convertir la variable objetivo en valores numéricos entre 0 y el número de clases menos 1
categorical_features = df.select_dtypes(exclude=[np.number]).columns
encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])

print(df.head())
