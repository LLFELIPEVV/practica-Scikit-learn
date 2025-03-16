"""
Guía de Búsqueda en Cuadrícula para Ajustar el Parámetro C en Regresión Logística
================================================================================

La búsqueda en cuadrícula es una técnica para encontrar la mejor combinación de parámetros
que maximice el desempeño de un modelo de machine learning. En este ejemplo, ajustamos el parámetro C
de un modelo de regresión logística. El parámetro C controla la regularización:
  - Valores altos de C reducen la penalización, permitiendo que el modelo se ajuste más a los datos de entrenamiento.
  - Valores bajos de C aumentan la regularización, limitando la complejidad del modelo.

Usaremos el dataset de iris y evaluaremos el desempeño del modelo (medido con la puntuación R²)
para distintos valores de C generados aleatoriamente alrededor del valor por defecto (C = 1).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. Modelo con Parámetros Predeterminados
# ----------------------------
# Cargamos el dataset de iris.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Creamos el modelo de regresión logística con un número alto de iteraciones para asegurar la convergencia.
logit = LogisticRegression(max_iter=10000)
logit.fit(X, y)
score_default = logit.score(X, y)
print("Modelo con C=1 (valor por defecto):")
print(logit)
print("Puntuación con parámetros predeterminados:", score_default)
# Con C = 1 (valor por defecto) se obtuvo una puntuación de aproximadamente 0.973.

# ----------------------------
# 2. Implementación de la Búsqueda en Cuadrícula para el Parámetro C
# ----------------------------
# Generamos 3000 valores aleatorios de C centrados en 1 (media = 1, desviación = 1) y seleccionamos solo los positivos.
C_values = np.random.normal(loc=1, scale=1, size=300)
C_values = C_values[C_values > 0]  # Solo consideramos valores positivos

# Inicializamos un diccionario para guardar las puntuaciones para cada valor de C,
# y variables para almacenar el mejor score y el mejor valor de C.
scores = {}
max_score = -np.inf  # Valor muy bajo inicial
best_C = None

# Iteramos sobre cada valor de C, ajustamos el modelo y evaluamos la puntuación.
for c in C_values:
    logit.set_params(C=c)
    logit.fit(X, y)
    score = logit.score(X, y)
    scores[c] = score
    if score > max_score:
        max_score = score
        best_C = c

print("\nMejor puntuación:", max_score, "con C:", best_C)
# Interpretación:
# Se encontró que un valor de C alrededor de 1.75 (por ejemplo) mejora ligeramente la precisión del modelo.
# Sin embargo, aumentar C más allá de este punto no aporta mejoras adicionales en la puntuación.

# (Opcional) Visualización de la relación entre C y la puntuación:
# Para entender cómo varía la puntuación con diferentes valores de C, podemos graficar los resultados.
plt.figure(figsize=(10, 6))
plt.scatter(list(scores.keys()), list(scores.values()),
            color="mediumseagreen", alpha=0.6, edgecolor="black")
plt.title("Relación entre el valor de C y la puntuación del modelo")
plt.xlabel("Valor de C")
plt.ylabel("Puntuación del modelo")
plt.axvline(x=best_C, color='red', linestyle='--',
            label=f"Mejor C ≈ {best_C:.2f}")
plt.legend()
plt.show()

if __name__ == "__main__":
    print("\n¡Finalizada la guía de búsqueda en cuadrícula para regresión logística!")
