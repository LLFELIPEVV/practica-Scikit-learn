import numpy as np
import matplotlib.pyplot as plt

# Aprendizaje automatico: entrenamiento y prueba
# Evalua tu modelo
# Para medir si el modelo es suficientemente bueno haciendo predicciones, podemos usar un metodo llamado entrenar/probar.
# Que es entrenar/probar
# Es un metodo para medir la presicion de un modelo.
# Se llama entrenar/probar por que divide el conjunto de datos en dos, un conjunto para entrenar y uno para probar.
# 80% para entrenamiento y el 20% para pruebas.
# Entrenar el modelo significa crear el modelo.
# Probar el modelo significa probar la presicion del modelo.

# Comenzando con un conjunto
# Comience con un conjunto de datos que desee probar.
# Nuestro conjunto de datos ilustra 100 clientes en una tienda y sus habitos de compra.
np.random.seed(2)

X = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / X

plt.scatter(X, y)
plt.show()

# Resultado
# El eje x representa el numero de minutos antes de realizar la compra.
# El eje y representa la cantidad de dinero gastado en la compra.

# Dividir el conjunto de datos
# El conjunto de entrenamiento debe ser una seccion aleatoria del 80% de los datos originales.
# El conjunto de pruebas debe ser el 20% restante.
train_x = X[:80]
train_y = y[:80]

test_x = X[80:]
test_y = y[80:]

# Mostrar el conjunto de entrenamiento
# Mostrar el mismo diagrama de dispersion con el conjunto de entrenamiento.
plt.scatter(train_x, train_y)
plt.show()
# Resultados
# Parece ser el conjunto de datos original por lo que es una seleccion justa.
