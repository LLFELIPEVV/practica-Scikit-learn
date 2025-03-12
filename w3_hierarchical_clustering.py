"""
Guía de Agrupamiento Jerárquico con Dendrograma y Visualización de Clusters
===========================================================================
La agrupación jerárquica es un método de aprendizaje no supervisado que agrupa puntos
de datos basándose en sus similitudes. En el agrupamiento aglomerativo, cada punto
comienza como su propio grupo y, de manera iterativa, se fusionan los grupos más cercanos,
hasta formar un único grupo global.

En este ejemplo:
  - Se crean dos listas 'x' e 'y' que representan dos variables.
  - Se combinan en un conjunto de puntos.
  - Se calcula la vinculación (linkage) usando la distancia euclidiana y el método de Ward,
    que minimiza la varianza dentro de cada grupo.
  - Se visualiza la jerarquía con un dendrograma.
  - Se utiliza AgglomerativeClustering de scikit-learn para segmentar los datos en 2 clusters,
    y se visualizan los clusters en un diagrama de dispersión.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Configuración de Seaborn para gráficos con estilo
sns.set_theme(style="whitegrid", font_scale=1.2)

# -------------------------------------------------------------------------
# Crear Datos Simulados
# -------------------------------------------------------------------------
# Dos listas que representan dos variables (por ejemplo, características de puntos)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combinar las listas en un conjunto de datos (lista de tuplas)
data = list(zip(x, y))

# -------------------------------------------------------------------------
# Calcular la Vinculación Jerárquica
# -------------------------------------------------------------------------
# Usamos la función linkage de SciPy con el método de Ward y la distancia euclidiana.
linked_data = linkage(data, method='ward', metric='euclidean')

# Visualizar la jerarquía de agrupamiento mediante un dendrograma.
plt.figure(figsize=(10, 6))
dendrogram(linked_data)
plt.title("Dendrograma del Agrupamiento Jerárquico")
plt.xlabel("Índice del Punto de Datos")
plt.ylabel("Distancia Euclidiana")
plt.show()

# -------------------------------------------------------------------------
# Agrupamiento Jerárquico con scikit-learn
# -------------------------------------------------------------------------
# Aplicamos AgglomerativeClustering para formar 2 clusters en base a los datos.
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

# Visualizar los clusters con un diagrama de dispersión.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, hue=labels, palette="viridis", s=100, edgecolor="black")
plt.title("Agrupamiento Jerárquico: Clusters")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.show()

# -------------------------------------------------------------------------
# Explicación:
# -------------------------------------------------------------------------
# 1. Se crean dos listas 'x' e 'y' que representan dos variables de un conjunto de datos.
# 2. Se combinan en una lista de puntos utilizando list(zip(x, y)).
# 3. Con la función linkage() se calcula la relación jerárquica entre los puntos
#    utilizando el método de Ward, que minimiza la varianza dentro de cada grupo.
# 4. El dendrograma resultante visualiza la jerarquía de agrupación, mostrando cómo se
#    unen los puntos desde grupos individuales hasta la formación de un único grupo.
# 5. Luego, usando AgglomerativeClustering, se segmentan los datos en 2 clusters, y se
#    visualizan los resultados en un diagrama de dispersión, donde cada color representa un cluster.
  
if __name__ == "__main__":
    print("¡Finalizada la guía de agrupamiento jerárquico!")
