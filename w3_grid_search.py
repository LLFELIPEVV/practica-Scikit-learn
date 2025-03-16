import numpy as np

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Busqueda de Cuadricula
# La mayoria de los modelos de aprendizaje automatico contienen parametros que pueden ajustarse para variar su aprendizaje.
# Por ejemplo el modelo de regresion logistica de sklearn tiene un parametro C que controla la regularizacion, lo cual afecta la complejidad del modelo.

# ¿Como funciona?
# Un metodo consiste en probar diferentes valores y combinaciones y seleccionar el que ofrezca la mayor puntuacion.
# Esta tecnica se conoce como busqueda en cuadricula.
# Si tuvieramos que seleccionar los valores de dos o mas parametros, evaluariamos todas la combinaciones de los conjuntos de valores, formando asi una cuadricula de valores.
# Valores altos de C le indican al modelo que los datos de entrenamiento se asemejan a la informacion del mundo real y le otorgan mayor importancia a estos. Valores de C tienen el efecto contrario.

# Uso de parametros predeterminados
# Observaremos primero los resultados generados sin una busqueda de cuadricula utilizando los parametros base.
# Se crea el modelo estableciendo el max_iter en un valor mas alto para garantizar que el modelo encuentre un resultado.
# Se debe tener en cuenta que el valor de C en un modelo de regresion logistica es 1.
iris = datasets.load_iris()

X = iris.data
y = iris.target

logit = LogisticRegression(max_iter=10000)
print(logit.fit(X, y))
print(logit.score(X, y))
# Con la configuracion predeterminada de C = 1, logramos una puntuacion de 0.973.

# Implementacion de la busqueda de cuadricula
# Ahora observaremos los resultados utilizando un rango de parametros de C.
# Como el valor predeterminado de C es 1 el rango se establecera con valores cercanos al 1.
C = np.array(np.random.normal(loc=1, scale=1, size=3000))
C = C[C > 0]

scores = {}
max_score = -np.inf  # Inicializamos con un valor muy bajo
best_choice = None

for c in C:
    logit.set_params(C=c)
    logit.fit(X, y)
    score = logit.score(X, y)
    scores[c] = score
    if score > max_score:
        max_score = score
        best_choice = c

print("Mejor score:", max_score, "con C:", best_choice)

# El valor de C en 1,75 obtuvo un aumento de presicion, sin embargo subirlo apartir de este no contribuye al aumento de presicion.
