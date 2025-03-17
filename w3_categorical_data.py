import pandas as pd

# Datos categoricos
# Cuando los datos tienen categorias representadas por cadenas, sera dificil usarlas para entrenar modelos de aprendizaje automatico que a menudo solo aceptan datos numericos.
# En lugar de ignorar los datos categoricos, se pueden transformar.
cars = pd.read_csv("data.csv")
print(cars.head())

# En regresion multiple tratamos de predecir el CO2 emitido por un vehiculo en funcion del motor y el peso, pero excluimos informacion sobre la marca y el modelo del auto.
