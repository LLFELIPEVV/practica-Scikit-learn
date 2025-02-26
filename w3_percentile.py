"""
Guía de Percentiles en Estadística para Machine Learning
=========================================================

Los percentiles son medidas estadísticas que nos indican el valor bajo el cual se
encuentra un determinado porcentaje de datos en un conjunto. Por ejemplo, el percentil
75 es el valor por debajo del cual se encuentra el 75% de los datos.

Ejemplo práctico:
Supongamos que tenemos una lista con las edades de las personas que viven en una calle.
Usaremos percentiles para responder preguntas como:
    - ¿Cuál es la edad que marca el 75% de la población? (es decir, el 75% tiene esa edad o menos)
    - ¿Cuál es la edad que marca el 90% de la población?
    
Para ello, usaremos la función np.percentile() de NumPy.
"""

import numpy as np

# Ejemplo: Lista de edades de personas en una calle
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39,
        80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]

# Calcular el percentil 75:
# Esto significa que el 75% de las edades son menores o iguales a este valor.
percentil_75 = np.percentile(ages, 75)
print("El percentil 75 (edad bajo la cual se encuentra el 75% de las personas) es:", percentil_75)
# En este caso, se espera que el resultado sea 43.

# Calcular el percentil 90:
# Esto nos dice que el 90% de las edades están por debajo de este valor.
percentil_90 = np.percentile(ages, 90)
print("El percentil 90 (edad bajo la cual se encuentra el 90% de las personas) es:", percentil_90)

# -----------------------------------------------------------------------------
# Resumen:
# -----------------------------------------------------------------------------
# - Los percentiles son útiles para entender la distribución de los datos.
# - np.percentile(lista, p) calcula el valor por debajo del cual se encuentra el p%
#   de los datos.
#
# Con estos conceptos, puedes determinar, por ejemplo, el rango de edad en el que se
# encuentra la mayoría de una población.
#
# ¡Utiliza estos conocimientos para analizar y preprocesar tus datos en proyectos de Machine Learning!
