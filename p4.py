import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, kruskal

df = pd.read_csv("uanl.csv")

df['fecha'] = pd.to_datetime({'year': df['anio'], 'month': df['mes'], 'day': 1})

print("Resumen estadístico general:")
print(df.describe(include='all'))

grouped = df.groupby('dependencia')['Sueldo Neto'].agg(['count', 'mean', 'std', 'min', 'max']).sort_values(by='count', ascending=False)
print("\nEstadísticas por dependencia:")
print(grouped.head(10))

deps = df['dependencia'].value_counts().index[:3]

samples = [df[df['dependencia'] == dep]['Sueldo Neto'].sample(100, random_state=1) for dep in deps]

anova_result = f_oneway(*samples)
print("Resultado ANOVA:", anova_result)

group1 = samples[0]
group2 = samples[1]
ttest_result = ttest_ind(group1, group2)
print("Resultado T-Test:", ttest_result)
kruskal_result = kruskal(*samples)
print("Resultado Kruskal-Wallis:", kruskal_result)
