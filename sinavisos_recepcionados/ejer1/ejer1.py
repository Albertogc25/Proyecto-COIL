import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def save_and_show_plot(filename):
    plt.savefig(filename)
    plt.show()
    plt.close()

# Leer el archivo CSV
df = pd.read_csv('/workspaces/Proyecto-COIL/sinavisos_recepcionados/aviso_recepcionados_limpio.csv', sep=',')

# Preprocesamiento: Codificar variables categóricas
le = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns

# Convertir las columnas con tipos mixtos a strings
for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, (float, int))).any() and df[col].apply(lambda x: isinstance(x, str)).any():
        df[col] = df[col].astype(str)

# Codificar las variables categóricas
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Manejar valores NaN
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Histogramas
for col in numeric_columns:
    plt.figure()
    df[col].plot.hist(title=f'Histograma de {col}', figsize=(10, 5))
    save_and_show_plot(f"histograma_{col}.png")

# Boxplots
for col in numeric_columns:
    plt.figure()
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot de {col}')
    save_and_show_plot(f"boxplot_{col}.png")

# Resto del código sin cambios
