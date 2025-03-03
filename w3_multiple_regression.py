import pandas as pd

from sklearn import linear_model

# La regresion multiple es como la regresion lineal pero usando mas de una variable para realizar predicciones.
# El modulo Pandas nos permite leer archivos csv devolver un objeto DataFrame.
df = pd.read_csv("data.csv")

# Luego se hace una lista de valores independientes, y se llama x.
# Y los valores dependientes se llaman y.
X = df[['Weight', 'Volume']]
y = df['CO2']

# Es comun nombrar la lista de valores independientes con una x mayuscula y la lista de valores dependientes con una y minuscula.
# Se usara el metodo LinearRegression() para crear un objeto de regresion lineal.
# Este objeto tiene un metodo llamado fit() que toma los valores independientes y dependientes como parametros y llenara el objeto de regresion lineal con datos que describan las relaciones entre los valores.
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Ahora tenemos un objeto que puede predecir los valores de CO2 en funcion del peso y el volumen de un auto.
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)
