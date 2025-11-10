import pandas as pd

df = pd.read_csv("uanl.csv")

df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})

print("Resumen estadístico general:")
print(df.describe(include='all'))

grouped = df.groupby('dependencia')['Sueldo Neto'].agg(['count', 'mean', 'std', 'min', 'max']).sort_values(by='count', ascending=False)
print("\nEstadísticas por dependencia:")
print(grouped.head(10))
