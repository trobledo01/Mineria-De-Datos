import pandas as pd

df = pd.read_csv("uanl.csv")

df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})
