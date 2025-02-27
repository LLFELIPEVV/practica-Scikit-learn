import numpy as np
import matplotlib.pyplot as plt

# Un diagrama de dispersion es un diagrama en el cual cada valor del cojunto de datos se representa con un punto individual.
# Matplotlib tiene un metodo para dibujar graficos de dispersion, en el cual necesita de dos matrices, uno para los valores en x y otro para los valores en y. Ambas matrices deben ser de la misma longitud.
# Ejemplo: Utilizar el scatter() metodo para dibujar un grafico de dispersion.
# Los valores en x representan la edad de los coches.
# Los valores en y representan la velocidad de los coches.
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

plt.scatter(x, y)
plt.show()

# Explicacion del diagrama de dispersion.
# Lo que se puede analizar del diagrama es que los dos autos mas rapidos tienen 2 años y el auto mas lento tiene 12 años.
# Aunque esto puede ser una coincidencia ya que solo analizamos 13 datos.

# Distribuciones de datos aleatorioas.
# Se pueden realizar distribuciones de datos aleatorias para probar el funcionamiento de algun algoritmo.
# Ejemplo: Un diagrama de dispersion con 1000 puntos.
# Crearemos dos matrices que contengan 1000 datos aleatorios de una distribucion de datos normal.
# La primera matriz tendra una media de 5 y una desviacion estandar de 1.
# La segunda matriz tendra una media de 10 y una desviacion estandar de 2.
x = np.random.normal(5, 1, 1000)
y = np.random.normal(10, 2, 1000)

plt.scatter(x, y)
plt.show()

# Explicacion del diagrama de dispersion
# Se puede observar que los datos se concentran alrededor del 5 en el eje x y 10 en el eje y.
# Tambien se observa que la dispersion es mayor en y que en x.
