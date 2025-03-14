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

# Coeficiente
# En la regresion logistica, el coeficiente representa el cambio esperado en las probabilidades logaritmicas de tener el resultado por unidad de cambio en X.
# Ejemplo:
log_odds = logr.coef_
odds = numpy.exp(log_odds)
print(odds)
# Esto indica que a medida que el tumor aumenta 1cm, las probabilidades de que sea un tumor canceroso aumentan en 4x.

# Probabilidad
# Los valores del coeficiente y la interseccion se pueden utilizar para encontrar la probabilidad de que cada tumor sea canceroso.
# Se crea una funcion que retorna un valor que representa la probabilidad de que la observacion dada sea un tumor.
def logit2prob(logr, x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


# Para encontrar las probabilidades logarítmicas de cada observación, primero debemos crear una fórmula que se parezca a la de la regresión lineal, extrayendo el coeficiente y la intersección.
# Para luego convertir las probabilidades logarítmicas en probabilidades debemos potenciar las probabilidades logarítmicas.
# Ahora que tenemos las probabilidades, podemos convertirlas en probabilidad dividiéndolas por 1 más las probabilidades.
# Utilicemos ahora la función con lo aprendido para averiguar la probabilidad de que cada tumor sea canceroso.
print(logit2prob(logr, X))

# Resultados explicados
# 3.78 0.61 La probabilidad de que un tumor con un tamaño de 3,78 cm sea canceroso es del 61%.
# 2.44 0.19 La probabilidad de que un tumor con un tamaño de 2,44 cm sea canceroso es del 19%.
# 2.09 0.13 La probabilidad de que un tumor con un tamaño de 2,09 cm sea canceroso es del 13%.
