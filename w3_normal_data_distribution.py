"""
Guía de la Distribución Normal (Gaussiana) para Machine Learning
================================================================

La distribución normal es una forma de organizar datos en la que la mayoría de los valores
se agrupan alrededor de un valor central (la media), y la frecuencia de los valores
disminuye a medida que se alejan de esta media. Esta distribución es conocida como la "curva
de campana" debido a su característica forma.

En este ejemplo, generamos 100,000 números aleatorios que siguen una distribución normal
con:
  - Media (valor promedio): 5.0
  - Desviación estándar: 1.0

La desviación estándar nos indica cuán dispersos están los datos alrededor de la media.
Un valor bajo de desviación estándar significa que los datos están muy concentrados cerca
de la media, mientras que un valor alto indica que los datos están más dispersos.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar Seaborn para mejorar la apariencia de los gráficos.
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------
# Generar datos con distribución normal (Gaussiana)
# -------------------------------------------------------------------
# np.random.normal(media, desviación_estándar, cantidad_de_valores)
datos = np.random.normal(5.0, 1.0, 100000)

# -------------------------------------------------------------------
# Visualización de la distribución usando un histograma
# -------------------------------------------------------------------
# El histograma mostrará cómo se distribuyen los 100,000 valores generados.
# Se utiliza 100 barras (bins) para representar la distribución.
plt.figure(figsize=(10, 6))
sns.histplot(datos, bins=100, color="mediumseagreen")
plt.title("Distribución Normal (Curva de Campana)")
plt.xlabel("Valores")
plt.ylabel("Frecuencia")
plt.show()

# -------------------------------------------------------------------
# Explicación:
# -------------------------------------------------------------------
# - Se generaron 100,000 números aleatorios que se concentran alrededor del valor 5.0,
#   pues ese es el valor medio definido.
# - La desviación estándar es 1.0, lo que significa que la mayoría de los datos estarán
#   aproximadamente entre 4.0 y 6.0 (es decir, a una distancia de 1 unidad de la media).
# - El histograma muestra la "curva de campana", donde se observa que a medida que
#   nos alejamos del 5, la cantidad de datos disminuye.
