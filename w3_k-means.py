"""
Guía de Clustering con K-Means
==============================

K-Means es un algoritmo de aprendizaje no supervisado que agrupa datos en K clusters.
El objetivo es minimizar la varianza dentro de cada cluster. En este ejemplo:
  1. Visualizamos un conjunto de puntos en 2D.
  2. Aplicamos el método del codo para determinar el número óptimo de clusters (K).
  3. Entrenamos K-Means con el valor óptimo de K.
  4. Visualizamos los resultados del clustering.

"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------------------------------------------------
# 1. Generación y Visualización de los Datos
# -------------------------------------------------------------------------
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.title("Puntos de datos originales")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Convertimos los datos en una lista de pares (x, y) para su procesamiento con K-Means
data = list(zip(x, y))

# -------------------------------------------------------------------------
# 2. Selección del Número de Clusters con el Método del Codo
# -------------------------------------------------------------------------
# K-Means requiere definir el número de clusters (K).
# El método del codo nos ayuda a encontrar un buen valor de K observando la inercia.

inertias = []  # Lista para almacenar la inercia de cada K

for i in range(1, 11):  # Probamos valores de K entre 1 y 10
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)  # Guardamos la inercia para este K

# Graficamos la inercia vs número de clusters para identificar el "codo"
plt.plot(range(1, 11), inertias, marker='o')
plt.title("Método del Codo")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inercia")
plt.show()

# Según la gráfica, se observa que K=2 es una buena elección (punto donde la inercia deja de disminuir abruptamente).

# -------------------------------------------------------------------------
# 3. Aplicación de K-Means con K=2 y Visualización del Resultado
# -------------------------------------------------------------------------
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(data)

# Mostramos los clusters encontrados mediante colores distintos
plt.scatter(x, y, c=kmeans.labels_, cmap='viridis', edgecolors='black')
plt.title("Resultados del Clustering con K-Means (K=2)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# -------------------------------------------------------------------------
# 4. Explicación del Proceso
# -------------------------------------------------------------------------
"""
- Se definieron los datos en un espacio bidimensional.
- Se utilizó el método del codo para encontrar el valor óptimo de K.
- Se entrenó K-Means con K=2.
- Se graficaron los resultados mostrando la agrupación de los puntos.

Este mismo procedimiento puede aplicarse a conjuntos de datos con cualquier número de dimensiones.
"""

if __name__ == "__main__":
    print("\n¡Finalizado el análisis de clustering con K-Means!")
