import numpy

from sklearn import linear_model

# Regresion logistica
# Busca resolver problemas de clasificacion. Esto lo hace prediciendo resultados categoricos, a diferencia de la regresion lineal, que predice un resultado continuo.
# En el caso mas simple, existen dos resultados, lo que se denomina binomial. Por ejemplo predecir si un tumor es benigno o maligno. En otros casos se requieren mas de dos resultados para clasificar; en este caso, se denomina multinomial. Un ejemplo de regresion logistica multinomial seria predecir la clase de una flor entre tres especies diferentes.
# Usemos el ejemplo de los tumores:
# X representa el tamaño del tumor en centimetros
# X debe tranformarse en una columna apartir de una fila para que la funcion LogisticRegression() funcione.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92,
                4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
# y representa si el tumor es canceroso o no.
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

# Predecir si el tumor es canceroso cuando su tamaño es 3.46cm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))
print(predicted)
