import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("uanl.csv")

df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})

print("Resumen estadístico general:")
print(df.describe(include='all'))

grouped = df.groupby('dependencia')['Sueldo Neto'].agg(['count', 'mean', 'std', 'min', 'max']).sort_values(by='count', ascending=False)
print("\nEstadísticas por dependencia:")
print(grouped.head(10))

# Top 10 dependencias por número de empleados
plt.figure(figsize=(10, 5))
df['dependencia'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 dependencias")
plt.xlabel("Número de empleados")
plt.ylabel("Dependencia")
plt.tight_layout()
plt.show()

# Histograma de Sueldo Neto
plt.figure()
df['Sueldo Neto'].hist(bins=50)
plt.title("Histograma de sueldos")
plt.xlabel("Sueldo Neto")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot de Sueldo Neto
plt.figure()
sns.boxplot(x='Sueldo Neto', data=df)
plt.title("Boxplot de sueldos")
plt.show()

# Gráfico de dispersión sueldo vs mes (muestra aleatoria para evitar sobrecarga)
plt.figure()
sns.scatterplot(x='mes', y='Sueldo Neto', hue='dependencia', data=df.sample(1000))
plt.title("Dispersión Sueldo vs Mes")
plt.show()

# Gráfico circular (pie chart) de distribución por mes
plt.figure()
df['mes'].value_counts().sort_index().plot.pie(autopct='%1.1f%%')
plt.title("Distribución de registros por mes")
plt.ylabel("")
plt.show()
