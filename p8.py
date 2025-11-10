import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("uanl.csv")
df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})

# Agrupar sueldo promedio por fecha
monthly_avg = df.groupby('fecha')['Sueldo Neto'].mean().reset_index()

# Convertir fechas a índices numéricos
X = np.arange(len(monthly_avg)).reshape(-1, 1)
y = monthly_avg['Sueldo Neto'].values

# Modelo de regresión lineal
reg = LinearRegression().fit(X, y)

# Predecir 12 meses futuros
future_X = np.arange(len(monthly_avg), len(monthly_avg)+12).reshape(-1, 1)
future_y = reg.predict(future_X)

# Graficar histórico y predicción
plt.figure()
plt.plot(monthly_avg['fecha'], y, label='Histórico')
future_dates = pd.date_range(start=monthly_avg['fecha'].iloc[-1], periods=13, freq='MS')[1:]
plt.plot(future_dates, future_y, label='Pronóstico', linestyle='--')
plt.title("Pronóstico del sueldo promedio")
plt.legend()
plt.show()
