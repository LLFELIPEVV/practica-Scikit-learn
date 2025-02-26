import numpy as np
import matplotlib.pyplot as plt
# Los conjuntos de datos suelen ser mucho mas grandes que con los que hemos trabajado, pero puede ser dificil recolectar datos del mundo real en una etapa temprana del proyecto.
# Para crear grandes conjuntos de datos para pruebas, se utiliza el modulo Numpy, que viene con varios metodos para crear conjuntos de datos aleatorios.

# Ejemplo: Crear una matriz que tenga 250 numeros flotantes aleatorios entre 0 y 5.
result = np.random.uniform(0, 5, 250)
print(result)

# Histograma
# Una forma de visualizar la distribucion de los datos es un histograma.
plt.hist(result, 5)
plt.show()

# Se usaron los datos aleatorios del ejemplo anterior para dibujar 5 barras en el histograma.
# La primera barra representa cuantos valores de la matriz estan entre 0 y 1.
# La segunda barra representa cuantos valores de la matriz estan entre 1 y 2.
# Y asi sucesivamente.
# El eje x representa el rango de valores.
# El eje y representa la frecuencia de valores en ese rango.
