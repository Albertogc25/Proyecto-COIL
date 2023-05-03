import pandas as pd

df = pd.read_csv('FicheroRuido2023.csv', sep=';', encoding='ISO-8859-1')

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.')

# Si necesitas convertir alguna columna a tipo numérico después de haber reemplazado las comas por puntos:
# df['nombre_columna'] = pd.to_numeric(df['nombre_columna'])

df.to_csv('nombre_archivo_limpio.csv', index=False)
