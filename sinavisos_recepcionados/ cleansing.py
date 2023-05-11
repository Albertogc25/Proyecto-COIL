# Importar las bibliotecas necesarias
import pandas as pd

# Leer el archivo CSV, utilizando el separador ';'
df = pd.read_csv("/workspaces/Proyecto-COIL/sinavisos_recepcionados/sic-avisos_recepcionados_2023.csv", sep=';', low_memory=False)

# Función para cambiar las ',' por '.' y convertir el resultado en float si es posible
def cambiar_coma_por_punto(valor):
    if isinstance(valor, str):
        try:
            return float(valor.replace(',', '.'))
        except ValueError:
            return valor
    return valor

# Aplicar la función a todas las columnas del DataFrame
for columna in df.columns:
    df[columna] = df[columna].apply(cambiar_coma_por_punto)

# Guardar el DataFrame en un nuevo archivo CSV con el separador ','
ruta_archivo_limpio = r"C:\Users\marie\Documents\GitHub\Proyecto-COIL\LIMPIEZA-SIC-CSV\sic-avisos_recepcionados_2023_limpio.csv"
df.to_csv(ruta_archivo_limpio, index=False, sep=',')
