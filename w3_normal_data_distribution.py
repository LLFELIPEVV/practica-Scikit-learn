import numpy as np
import matplotlib.pyplot as plt

# La distribucion de datos normal permite crear una matriz donde los valores se concentran alrededor de un valor especifico.
# Esta distribucion se llama normal o Gaussiana.
# Ejemplo: Hacer una distribucion de datos normal tipica.
result = np.random.normal(5.0, 1.0, 100000)
plt.hist(result, 100)
plt.show()
# El grafico de una distribucion normal tambien se conoce como curva de campana debido a su caracteristica forma de campana.

# Explicacion del histograma
# Se crearon 100000 valores randoms en el cual el valor medio seria 5 y la desviacion estandar seria 1, lo que significa que los valores debian concentrarse alrededor del 5 y no alejarse mas de 1 unidad de la media.
