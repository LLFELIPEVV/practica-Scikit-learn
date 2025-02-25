"""
Guía de Desviación Estándar y Varianza para Machine Learning
=============================================================

La desviación estándar es una medida que indica qué tan dispersos están los valores
de un conjunto de datos respecto a la media. Una desviación estándar baja significa
que la mayoría de los datos están cerca del promedio, mientras que una alta indica 
mayor dispersión.

La varianza es otra medida de dispersión y se calcula como la media de los valores 
al cuadrado de la diferencia de cada dato con respecto a la media. La desviación 
estándar es simplemente la raíz cuadrada de la varianza.

Además, en notación:
- La desviación estándar se representa con la letra griega sigma (σ).
- La varianza se representa con sigma al cuadrado (σ²).

A continuación se muestran ejemplos prácticos usando Python y NumPy.
"""

import numpy as np

# -------------------------------
# Ejemplo 1: Datos con baja dispersión
# -------------------------------
# Imagina que registramos la velocidad (en km/h) de 7 coches, donde los datos
# son bastante similares:
speed1 = [86, 87, 88, 86, 87, 85, 86]
# La media aproximada es 86.4, lo que significa que la mayoría de los valores
# están cerca de este promedio. Por ello, la desviación estándar es baja.
std_speed1 = np.std(speed1)
print("Desviación estándar de speed1 (baja dispersión):", std_speed1)
# Esperamos un valor cercano a 0.9

# -------------------------------
# Ejemplo 2: Datos con alta dispersión
# -------------------------------
# Ahora consideremos otro conjunto de velocidades, donde los datos varían mucho:
speed2 = [32, 111, 138, 28, 59, 77, 97]
# La media aproximada es 77.3, pero los valores están muy dispersos.
# Esto se refleja en una desviación estándar alta.
std_speed2 = np.std(speed2)
print("Desviación estándar de speed2 (alta dispersión):", std_speed2)
# Se espera un valor alrededor de 37.85

# -------------------------------
# Cálculo Manual de la Varianza para 'speed2'
# -------------------------------
# La varianza es el promedio de los cuadrados de las diferencias de cada valor con la media.
# Paso 1: Calcular la media de speed2
media_speed2 = sum(speed2) / len(speed2)

# Paso 2: Calcular la diferencia de cada valor con la media
diferencias = [(x - media_speed2) for x in speed2]

# Paso 3: Elevar cada diferencia al cuadrado
cuadrados = [d ** 2 for d in diferencias]

# Paso 4: Calcular la media de esos valores cuadrados (esto es la varianza)
varianza_manual = sum(cuadrados) / len(cuadrados)
print("Varianza calculada manualmente para speed2:", varianza_manual)

# -------------------------------
# Cálculo de la Varianza usando NumPy
# -------------------------------
# NumPy tiene la función np.var() que calcula la varianza de forma directa.
varianza_numpy = np.var(speed2)
print("Varianza usando numpy para speed2:", varianza_numpy)

# Notas:
# - La desviación estándar (σ) nos indica cuánto se alejan en promedio los datos de la media.
# - La varianza (σ²) es el cuadrado de la desviación estándar y nos da una medida de dispersión
#   en unidades al cuadrado.
