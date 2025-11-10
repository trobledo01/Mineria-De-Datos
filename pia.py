
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import f_oneway, ttest_ind, kruskal
import numpy as np


# Data Cleaning
# ==========================


df = pd.read_csv("uanl.csv")

df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})


# Descriptive Statistics
# ==========================

print("Resumen estadístico :")
print(df.describe(include='all'))

grouped = df.groupby('dependencia')['Sueldo Neto'].agg(['count', 'mean', 'std', 'min', 'max']).sort_values(by='count', ascending=False)
print("\\nEstadísticas por dependencia:")
print(grouped.head(10))


# Data Visualization
# ==========================

plt.figure(figsize=(10, 5))
df['dependencia'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 dependencias")
plt.xlabel("Número de empleados")
plt.ylabel("Dependencia")
plt.tight_layout()
plt.show()

plt.figure()
df['Sueldo Neto'].hist(bins=50)
plt.title("Histograma de sueldos")
plt.xlabel("Sueldo Neto")
plt.ylabel("Frecuencia")
plt.show()

plt.figure()
sns.boxplot(x='Sueldo Neto', data=df)
plt.title("Boxplot de sueldos")
plt.show()

#plt.figure()
#sns.scatterplot(x='mes', y='Sueldo Neto', hue='dependencia', data=df.sample(1000))
#plt.title("Dispersión sueldo vs mes")
#plt.show()


# 1. Ajusta el tamaño de la figura para que haya espacio
plt.figure(figsize=(10, 7)) # Puedes probar (12, 8) si es muy larga

sns.scatterplot(x='mes', y='Sueldo Neto', hue='dependencia', data=df.sample(1000))
plt.title("Dispersión sueldo vs mes")

# --- AQUÍ LA SOLUCIÓN ---
# 2. Mueve la leyenda fuera del gráfico
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# 3. Ajusta el margen derecho para que la leyenda no se corte
#    (Puedes jugar con el valor 0.6, 0.7, 0.5)
plt.subplots_adjust(right=0.6)

plt.show()

plt.figure()
df['mes'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Distribución por mes")
plt.ylabel("")
plt.show()

# Statistic Test (ANOVA + T-Test / Kruskal-Wallis)
# ==========================


deps = df['dependencia'].value_counts().index[:3]
samples = [df[df['dependencia'] == dep]['Sueldo Neto'].sample(100) for dep in deps]
anova_result = f_oneway(*samples)
print("\\nANOVA result:", anova_result)

group1 = df[df['dependencia'] == deps[0]]['Sueldo Neto'].sample(100)
group2 = df[df['dependencia'] == deps[1]]['Sueldo Neto'].sample(100)
ttest_result = ttest_ind(group1, group2)
print("T-test result:", ttest_result)

kruskal_result = kruskal(*samples)
print("Kruskal-Wallis result:", kruskal_result)

#  Linear Model + Correlation
# ==========================

df['tiempo'] = df['anio'] * 12 + df['mes']
X = df[['tiempo']]
y = df['Sueldo Neto']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("\\nR2 Score:", r2)

plt.figure()
plt.scatter(X, y, alpha=0.1, label='Datos')
plt.plot(X, y_pred, color='red', label='Modelo lineal')
plt.title("Modelo Lineal Sueldo vs Tiempo")
plt.legend()
plt.show()

# Data Classification (KNN)
# ==========================


df_knn = df[df['dependencia'].isin(deps)]
X = df_knn[['mes', 'anio']]
y = df_knn['dependencia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("\\nPrecisión KNN:", knn.score(X_test, y_test))

#  Data Clustering (KMeans)
# ==========================

kmeans = KMeans(n_clusters=3, n_init=10)
X_kmeans = df[['Sueldo Neto', 'mes']]
kmeans.fit(X_kmeans)
df['cluster'] = kmeans.labels_

plt.figure()
sns.scatterplot(data=df.sample(1000), x='Sueldo Neto', y='mes', hue='cluster', palette='Set2')
plt.title("K-Means Clustering")
plt.show()

#  Forecasting (Time Series Linear Regression)
# ==========================

monthly_avg = df.groupby('fecha')['Sueldo Neto'].mean().reset_index()
X = np.arange(len(monthly_avg)).reshape(-1, 1)
y = monthly_avg['Sueldo Neto'].values
reg = LinearRegression().fit(X, y)
future_X = np.arange(len(monthly_avg), len(monthly_avg)+12).reshape(-1, 1)
future_y = reg.predict(future_X)

plt.figure()
plt.plot(monthly_avg['fecha'], y, label='Histórico')
plt.plot(pd.date_range(monthly_avg['fecha'].iloc[-1], periods=13, freq='MS')[1:], future_y, label='Pronóstico', linestyle='--')
plt.legend()
plt.title("Pronóstico Sueldo Neto Promedio")
plt.show()

#  Text Analysis (Word Cloud)
# ==========================

text = " ".join(df['Nombre'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud de Nombres")
plt.show()