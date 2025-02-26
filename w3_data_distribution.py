"""
Guía para generar y visualizar conjuntos de datos grandes usando NumPy y Seaborn
================================================================================

En Machine Learning, a menudo se requiere probar algoritmos con grandes conjuntos de datos.
Para ello, podemos generar datos aleatorios usando NumPy y visualizar su distribución mediante
histogramas. En este ejemplo, se muestran dos casos:

1. Un conjunto pequeño de 250 valores aleatorios entre 0 y 5, visualizado con 5 barras.
2. Un conjunto grande de 1,000,000 de valores aleatorios entre 0 y 90, visualizado con 100 barras.

Se utilizan Seaborn y Matplotlib para crear gráficos con una estética mejorada.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo de Seaborn para mejorar la estética de los gráficos.
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# Ejemplo 1: Conjunto pequeño de datos
# -----------------------------------------------------------------------------
# Crear un arreglo de 250 números flotantes aleatorios entre 0 y 5.
data_small = np.random.uniform(0, 5, 250)
print("Conjunto pequeño (250 valores):")
print(data_small)

# Visualizar la distribución de 'data_small' con un histograma de 5 barras.
plt.figure(figsize=(8, 4))
sns.histplot(data_small, bins=5, kde=False, color="skyblue")
plt.title("Histograma: 250 valores aleatorios entre 0 y 5")
plt.xlabel("Rango de valores")
plt.ylabel("Frecuencia")
plt.show()

# -----------------------------------------------------------------------------
# Ejemplo 2: Conjunto grande de datos
# -----------------------------------------------------------------------------
# Crear un arreglo de 1,000,000 de números flotantes aleatorios entre 0 y 90.
data_large = np.random.uniform(0, 90, 1000000)
print("\nTamaño del conjunto grande:", len(data_large))

# Visualizar la distribución de 'data_large' con un histograma de 100 barras.
plt.figure(figsize=(10, 6))
sns.histplot(data_large, bins=100, kde=False, color="salmon")
plt.title("Histograma: 1,000,000 valores aleatorios entre 0 y 90")
plt.xlabel("Rango de valores")
plt.ylabel("Frecuencia")
plt.show()

if __name__ == "__main__":
    print("Visualización completada.")
