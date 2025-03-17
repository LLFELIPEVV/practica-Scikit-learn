import pandas as pd

from sklearn import linear_model

# Datos categoricos
# Cuando los datos tienen categorias representadas por cadenas, sera dificil usarlas para entrenar modelos de aprendizaje automatico que a menudo solo aceptan datos numericos.
# En lugar de ignorar los datos categoricos, se pueden transformar.
cars = pd.read_csv("data.csv")
print(cars.head())

# En regresion multiple tratamos de predecir el CO2 emitido por un vehiculo en funcion del motor y el peso, pero excluimos informacion sobre la marca y el modelo del auto.

# Codificacion One Hot
# No se puede utilizar la columna Coche ni la columna Modelo en nuestros datos, ya que no son numericas. No se puede determinar una relacion lineal entre una variable categorica y una variable numerica.
# Para solucionar este problema, necesitamos una representacion numerica de la variable categorica. Una forma de lograrlo es tener una columna que represente cada grupo de la categoria.
# Para cada columna, los valores seran 1 o 0, donde 1 representa la inclusion del grupo y 0 la exclusion. Esta transformacion se denomina codificacion one-hot.
ohe_cars = pd.get_dummies(cars[["Car"]])
print(ohe_cars.head())

# Predecir el CO2
X = pd.concat([cars[["Volume", "Weight"]], ohe_cars], axis=1)
y = cars["CO2"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict(
    [[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

print(predictedCO2)
