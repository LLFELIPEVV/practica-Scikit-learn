import numpy as np

# La desviacion estandar es un numero que describe que tan dispersos estan los valores.
# Una desviacion estandar baja indica que los valores estan cerca de la media de los datos.
# Una desviacion estandar alta indica que los valores estan alejados de la media de los datos.
# Ejemplo: Registros de la velocidad de 7 coches
speed = [86, 87, 88, 86, 87, 85, 86]
# La desviacion estandar es 0.9
# Lo que siginifica que la mayoria de los valores estan dentro un rango de 0,9 respecto al valor promedio que es 86,4.
speed = [32, 111, 138, 28, 59, 77, 97]
# La desviacion estandar es 37,85
# Lo que siginifica que la mayoria de los valores estan dentro un rango de 37,85 respecto al valor promedio que es 77,3.

# Numpy tiene un metodo para calcular la desviacion estandar
speed = [86, 87, 88, 86, 87, 85, 86]
result = np.std(speed)
print(result)

speed = [32, 111, 138, 28, 59, 77, 97]
result = np.std(speed)
print(result)

# Varianza
# La varianza es otra medida de dispersión. Se calcula como la media de los valores al cuadrado.
# La varianza es otro numero que indica que tan dispersos estan los valores.
# La raiz cuadrada de la varianza de como resultado la desviacion estandar.
# El cuadrado de la desviacion estandar es la varianza.
# Paso para calcular la varianza:
# Encontrar la media
result = sum(speed) / len(speed)
# Encontrar la diferencia con la media
diferencias = [(x - result) for x in speed]
# Encontrar el valor cuadrado
cuadrados = [x**2 for x in diferencias]
result = sum(cuadrados) / len(cuadrados)
print(result)

# Numpy tiene un metodo para calcular la varianza
speed = [32, 111, 138, 28, 59, 77, 97]
result = np.var(speed)
print(result)
