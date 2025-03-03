"""
Guía de Regresión Múltiple para Machine Learning
================================================

La regresión múltiple es una extensión de la regresión lineal que utiliza más de una variable 
independiente para predecir una variable dependiente. En este ejemplo, se usa un archivo CSV 
("data.csv") que contiene información sobre vehículos:
  - Weight: Peso del vehículo.
  - Volume: Volumen (por ejemplo, del motor o interior).
  - CO2: Emisiones de CO2.

Utilizaremos Pandas para leer los datos y el módulo linear_model de scikit‑learn para 
ajustar un modelo de regresión lineal múltiple. Este modelo nos permitirá predecir las emisiones 
de CO2 de un vehículo en función de su peso y volumen.
"""

import pandas as pd
from sklearn import linear_model

# Leer el archivo CSV y crear un DataFrame.
# Nota: Asegúrate de que "data.csv" esté en el mismo directorio que este archivo o proporciona la ruta correcta.
df = pd.read_csv("data.csv")

# Mostrar las primeras filas del DataFrame para comprender su estructura.
print("Primeras filas del DataFrame:")
print(df.head())

# Definir las variables independientes y la variable dependiente:
# - X contiene las columnas 'Weight' y 'Volume' (variables que usaremos para predecir).
# - y contiene la columna 'CO2' (la variable que queremos predecir).
X = df[['Weight', 'Volume']]
y = df['CO2']

# Crear un objeto de regresión lineal.
regr = linear_model.LinearRegression()

# Ajustar (entrenar) el modelo con los datos.
# Esto encuentra la relación entre Weight, Volume y CO2.
regr.fit(X, y)

# Una vez ajustado el modelo, podemos usarlo para hacer predicciones.

# Ejemplo 1: Predecir las emisiones de CO2 para un vehículo con:
#   Peso: 2300 kg
#   Volumen: 1300 cm³
predictedCO2_case1 = regr.predict([[2300, 1300]])
print("\nEmisiones de CO2 predichas para un vehículo con 2300 kg y 1300 cm³:")
print(predictedCO2_case1)

# Imprimir los coeficientes del modelo.
# Los coeficientes indican el cambio en las emisiones de CO2 por cada unidad de cambio en cada variable.
# Por ejemplo, un coeficiente de 0.00755 para 'Weight' significa que por cada kg adicional,
# las emisiones de CO2 aumentan en 0.00755 unidades (según la escala de los datos).
print("\nCoeficientes del modelo (Weight y Volume):")
print(regr.coef_)

# Ejemplo 2: Predecir las emisiones de CO2 para un vehículo con:
#   Peso: 3300 kg
#   Volumen: 1300 cm³
predictedCO2_case2 = regr.predict([[3300, 1300]])
print("\nEmisiones de CO2 predichas para un vehículo con 3300 kg y 1300 cm³:")
print(predictedCO2_case2)

if __name__ == "__main__":
    print("\n¡Finalizada la guía de regresión múltiple!")
