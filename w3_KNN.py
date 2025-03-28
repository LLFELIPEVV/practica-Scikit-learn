import matplotlib.pyplot as plt

# KNN es un algoritmo de aprendizaje automatico supervisado.
# K es el numero de vecinos mas cercanos que se utilizaran para la clasificacion.
# Para la clasificacion se utiliza una mayoria de votos para determinar la clase a la que debe pertenecer una nueva observacion.
# Valores mayores de K suelen ser mas robustos a los valores atipicos y producen limites de decision mas estables que valores muy pequeños.
x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

plt.scatter(x, y, c=classes)
plt.show()
