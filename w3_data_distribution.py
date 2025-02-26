import numpy as np
# Los conjuntos de datos suelen ser mucho mas grandes que con los que hemos trabajado, pero puede ser dificil recolectar datos del mundo real en una etapa temprana del proyecto.
# Para crear grandes conjuntos de datos para pruebas, se utiliza el modulo Numpy, que viene con varios metodos para crear conjuntos de datos aleatorios.

# Ejemplo: Crear una matriz que tenga 250 numeros flotantes aleatorios entre 0 y 5.
result = np.random.uniform(0, 5, 250)
print(result)
