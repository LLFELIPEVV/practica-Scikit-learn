"""
Guía de Estadística Básica para Machine Learning
=================================================

En Machine Learning es fundamental entender cómo se distribuyen nuestros datos. 
Las tres medidas estadísticas más comunes son:

    - Media: El promedio de todos los valores.
    - Mediana: El valor que se encuentra en el centro de un conjunto de datos ordenados.
    - Moda: El valor que aparece con mayor frecuencia.

En este ejemplo, trabajaremos con un conjunto de datos que representa la velocidad (en km/h)
de 13 coches.
"""

import numpy as np
from scipy import stats

# Ejemplo: Velocidades registradas de 13 coches
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
print("Conjunto de datos (speed):", speed)

# -----------------------
# Cálculo de la Media
# -----------------------
# La media se obtiene sumando todos los valores y dividiendo entre el número total de elementos.
media_manual = sum(speed) / len(speed)
print("\nMedia (calculada manualmente):", media_manual)

# Numpy ofrece una función para calcular la media:
media_numpy = np.mean(speed)
print("Media (usando numpy):", media_numpy)

# -----------------------
# Cálculo de la Mediana
# -----------------------
# La mediana es el valor central en un conjunto de datos ordenados.

# Primero, ordenamos la lista sin modificar la original usando sorted()
speed_ordenada = sorted(speed)
print("\nDatos ordenados:", speed_ordenada)

# Caso 1: Número impar de elementos
# El elemento del medio se encuentra en la posición len(speed)//2
indice_medio = len(speed_ordenada) // 2
mediana_manual = speed_ordenada[indice_medio]
print("Mediana (número impar, calculada manualmente):", mediana_manual)

# Numpy tiene la función np.median() que lo calcula automáticamente:
mediana_numpy = np.median(speed)
print("Mediana (usando numpy):", mediana_numpy)

# Caso 2: Número par de elementos
# Tomamos un ejemplo con 12 elementos (quitamos uno del original)
speed_par = [99, 86, 87, 88, 86, 103, 87, 94, 78, 77, 85, 86]
speed_par_ordenada = sorted(speed_par)
print("\nDatos ordenados (número par):", speed_par_ordenada)

# Para datos pares, la mediana es el promedio de los dos valores centrales:
indice_medio_par = len(speed_par_ordenada) // 2
mediana_manual_par = (
    speed_par_ordenada[indice_medio_par - 1] + speed_par_ordenada[indice_medio_par]) / 2
print("Mediana (número par, calculada manualmente):", mediana_manual_par)

# Con numpy:
mediana_numpy_par = np.median(speed_par)
print("Mediana (usando numpy) con datos pares:", mediana_numpy_par)

# -----------------------
# Cálculo de la Moda
# -----------------------
# La moda es el valor que más se repite en el conjunto de datos.
# Con scipy.stats podemos calcularla fácilmente:
moda_resultado = stats.mode(speed, keepdims=True)
print("\nModa (usando scipy.stats):")
print("Valor de la moda:", moda_resultado.mode[0])
print("Frecuencia de la moda:", moda_resultado.count[0])

# -----------------------------------------------------------------------------
# Resumen:
# -----------------------------------------------------------------------------
# - La media nos proporciona el valor promedio de los datos.
# - La mediana indica el valor central de la distribución.
# - La moda es el valor que ocurre con mayor frecuencia.
#
# Estas medidas son esenciales para analizar la distribución de los datos y se usan
# ampliamente en algoritmos de Machine Learning para entender y preprocesar los datos.
#

if __name__ == "__main__":
    print("\n¡Finalizada la guía básica de medidas estadísticas!")
