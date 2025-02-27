"""
Guía de Diagramas de Dispersión (Scatter Plots) para Machine Learning
=====================================================================

Un diagrama de dispersión es un gráfico en el cual cada dato de un conjunto se representa
como un punto. Para crearlo, necesitamos dos conjuntos de valores (o listas/arrays):
uno para el eje x y otro para el eje y. Ambos deben tener la misma cantidad de elementos.

A continuación, veremos dos ejemplos:

1. Diagrama de dispersión con datos fijos:
   - En este ejemplo, los valores de "x" representan la edad de los coches (en años) y 
     los valores de "y" representan la velocidad de los coches (en km/h).
   - Observamos, por ejemplo, que los dos coches más rápidos tienen 2 años y el coche 
     más lento tiene 12 años (aunque son solo 13 datos, por lo que podría ser una coincidencia).

2. Diagrama de dispersión con datos aleatorios:
   - Generaremos dos arrays de 1000 números aleatorios usando una distribución normal.
   - El primer array (x) tendrá valores con una media de 5 y una desviación estándar de 1.
   - El segundo array (y) tendrá valores con una media de 10 y una desviación estándar de 2.
   - Así, veremos que los datos se concentran alrededor de 5 en el eje x y 10 en el eje y, 
     y que la dispersión es mayor en y que en x.

Utilizaremos Seaborn para mejorar la estética de los gráficos junto a Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de Seaborn para un estilo visual agradable
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------------
# Ejemplo 1: Diagrama de Dispersión con Datos Fijos
# -------------------------------------------------------------------------
# Los valores de x representan la edad de los coches (en años)
# Los valores de y representan la velocidad de los coches (en km/h)
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, s=100, color="dodgerblue", edgecolor="black")
plt.title("Diagrama de Dispersión: Edad vs. Velocidad de Coches")
plt.xlabel("Edad del coche (años)")
plt.ylabel("Velocidad del coche (km/h)")
plt.show()

# Explicación:
# Cada punto en el gráfico representa un coche. Se puede observar, por ejemplo, que los dos
# coches más rápidos (velocidades altas) tienen una edad de 2 años, mientras que el coche con
# la velocidad más baja (77 km/h) tiene 12 años. Cabe destacar que, al tratarse de solo 13 datos,
# estas observaciones pueden ser meramente coincidentes.

# -------------------------------------------------------------------------
# Ejemplo 2: Diagrama de Dispersión con Datos Aleatorios
# -------------------------------------------------------------------------
# Generaremos dos arrays de 1000 datos utilizando una distribución normal:
# - El array 'x_random' tendrá una media de 5 y una desviación estándar de 1.
# - El array 'y_random' tendrá una media de 10 y una desviación estándar de 2.

x_random = np.random.normal(5, 1, 1000)
y_random = np.random.normal(10, 2, 1000)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_random, y=y_random, s=40,
                color="salmon", alpha=0.6, edgecolor="black")
plt.title("Diagrama de Dispersión con Datos Aleatorios")
plt.xlabel("Valores en el eje X (media ~5)")
plt.ylabel("Valores en el eje Y (media ~10)")
plt.show()

# Explicación:
# En este gráfico, se observa que la mayoría de los datos se agrupan alrededor de 5 en el eje x
# y alrededor de 10 en el eje y, lo que concuerda con las medias definidas para cada distribución.
# Además, se nota que la dispersión (variabilidad) es mayor en el eje y (debido a una desviación
# estándar de 2) que en el eje x (desviación estándar de 1).

if __name__ == "__main__":
    print("¡Finalizada la guía de diagramas de dispersión!")
