"""
Clasificación con K-Nearest Neighbors (KNN)
===========================================

KNN es un algoritmo de clasificación supervisado basado en la vecindad de los puntos.
  - K representa el número de vecinos más cercanos utilizados para predecir una clase.
  - Valores más altos de K generan decisiones más estables y menos sensibles a valores atípicos.

En este script, se ilustra cómo cambia la predicción de un nuevo punto al ajustar el valor de K.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Configuración de estilo
sns.set(style="whitegrid", font_scale=1.2)

# Datos de ejemplo
x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Visualización inicial
plt.figure(figsize=(6, 5))
plt.scatter(x, y, c=classes, cmap="coolwarm", edgecolors="k", s=100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Distribución de datos con clases")
plt.colorbar(label="Clase")
plt.show()

# -----------------------------------------------------------------------------
# Modelo KNN con K = 1
# -----------------------------------------------------------------------------
data = list(zip(x, y))
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(data, classes)

new_x, new_y = 8, 21
new_point = [(new_x, new_y)]

prediction_1 = knn_1.predict(new_point)[0]

# Visualización de la predicción
plt.figure(figsize=(6, 5))
plt.scatter(x + [new_x], y + [new_y], c=classes +
            [prediction_1], cmap="coolwarm", edgecolors="k", s=100)
plt.text(new_x - 1.7, new_y - 0.7,
         s=f"Nuevo punto, Clase: {prediction_1}", fontsize=12, color="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clasificación con K=1")
plt.colorbar(label="Clase")
plt.show()

# -----------------------------------------------------------------------------
# Modelo KNN con K = 5
# -----------------------------------------------------------------------------
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(data, classes)

prediction_5 = knn_5.predict(new_point)[0]

# Visualización de la predicción con K=5
plt.figure(figsize=(6, 5))
plt.scatter(x + [new_x], y + [new_y], c=classes +
            [prediction_5], cmap="coolwarm", edgecolors="k", s=100)
plt.text(new_x - 1.7, new_y - 0.7,
         s=f"Nuevo punto, Clase: {prediction_5}", fontsize=12, color="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clasificación con K=5")
plt.colorbar(label="Clase")
plt.show()

# -----------------------------------------------------------------------------
# Explicación
# -----------------------------------------------------------------------------
"""
Cuando K=1, la clasificación depende únicamente del vecino más cercano.
  - En este caso, el vecino más próximo pertenece a la clase 0, por lo que el nuevo punto es clasificado como 0.

Cuando K=5, la clasificación depende de los cinco vecinos más cercanos.
  - Aquí, la mayoría de los vecinos pertenecen a la clase 1, lo que cambia la clasificación del nuevo punto a 1.
"""
