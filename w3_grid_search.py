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
