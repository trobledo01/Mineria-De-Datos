import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("uanl.csv")

# Clustering por Sueldo Neto y Mes
X_kmeans = df[['Sueldo Neto', 'mes']]
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
kmeans.fit(X_kmeans)

# Etiquetas de cluster
df['cluster'] = kmeans.labels_

# Visualización
plt.figure()
sns.scatterplot(data=df.sample(1000), x='Sueldo Neto', y='mes', hue='cluster', palette='Set2')
plt.title("Clustering con K-Means")
plt.show()
