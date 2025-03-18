"""
Guía de Preprocesamiento: Codificación One-Hot para Datos Categóricos
======================================================================

Cuando se trabaja con modelos de machine learning, es común encontrar datos
categóricos (por ejemplo, la marca o modelo de un auto) representados como cadenas de texto.
La mayoría de los algoritmos requieren datos numéricos, por lo que debemos transformar
estas variables. Una técnica común es la codificación One-Hot, que convierte cada categoría
en una columna binaria (0 o 1).

En este ejemplo:
  1. Leemos un archivo CSV ("data.csv") que contiene información de vehículos.
  2. Aplicamos la codificación One-Hot a la columna "Car".
  3. Preparamos las variables para predecir las emisiones de CO2 a partir de "Volume", "Weight"
     y la codificación de "Car".
  4. Entrenamos un modelo de regresión lineal múltiple.
  5. Realizamos una predicción de CO2.
  6. Mostramos ejemplos de codificación One-Hot usando la opción drop_first para reducir la redundancia.
"""

import pandas as pd
from sklearn import linear_model

# -------------------------------------------------------------------------
# 1. Lectura de Datos y Visualización Inicial
# -------------------------------------------------------------------------
# Leemos el archivo CSV "data.csv" que contiene información de vehículos.
cars = pd.read_csv("data.csv")
print("Primeras filas del DataFrame 'cars':")
print(cars.head())

# -------------------------------------------------------------------------
# 2. Codificación One-Hot para la Variable Categórica "Car"
# -------------------------------------------------------------------------
# La columna "Car" contiene datos no numéricos (por ejemplo, la marca o modelo del auto).
# Para usar estos datos en modelos de regresión, los transformamos en columnas numéricas
# utilizando pd.get_dummies(), que crea una columna binaria para cada categoría.
ohe_cars = pd.get_dummies(cars[["Car"]])
print("\nCodificación One-Hot de la columna 'Car':")
print(ohe_cars.head())

# -------------------------------------------------------------------------
# 3. Preparación de Datos para Predecir CO2
# -------------------------------------------------------------------------
# En regresión múltiple queremos predecir el CO2 emitido en función del motor y el peso del auto.
# Excluiremos la información de la marca/modelo ("Car") de forma directa, pero la incluiremos
# mediante la codificación One-Hot.
X = pd.concat([cars[["Volume", "Weight"]], ohe_cars], axis=1)
y = cars["CO2"]

print("\nCaracterísticas (X) para la predicción:")
print(X.head())
print("\nVariable objetivo (y) - CO2:")
print(y.head())

# -------------------------------------------------------------------------
# 4. Creación y Entrenamiento del Modelo de Regresión Múltiple
# -------------------------------------------------------------------------
regr = linear_model.LinearRegression()
regr.fit(X, y)

# -------------------------------------------------------------------------
# 5. Predicción de Emisiones de CO2
# -------------------------------------------------------------------------
# Predecimos las emisiones de CO2 para un vehículo con las siguientes características:
#   - Weight: 2300
#   - Volume: 1300
#   - Los valores One-Hot para la columna "Car" deben estar en el mismo orden que X.
predictedCO2 = regr.predict(
    [[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
print("\nEmisiones de CO2 predichas para el vehículo:", predictedCO2)

# -------------------------------------------------------------------------
# 6. Ejemplos de Codificación One-Hot con drop_first (Dummifying)
# -------------------------------------------------------------------------
# Con drop_first=True, se elimina una columna para evitar la redundancia.
# Ejemplo 1: Con dos categorías: 'blue' y 'red'
colors = pd.DataFrame({'color': ['blue', 'red']})
dummies = pd.get_dummies(colors, drop_first=True)
print("\nCodificación One-Hot (drop_first=True) para dos colores:")
print(dummies)
# En este caso, si 'blue' se codifica como 0 y 'red' como 1, basta con una columna.

# Ejemplo 2: Con tres categorías: 'blue', 'red' y 'green'
colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
dummies = pd.get_dummies(colors, drop_first=True)
# Si deseamos conservar la información original, podemos agregar la columna original.
dummies['color'] = colors['color']
print("\nCodificación One-Hot (drop_first=True) para tres colores:")
print(dummies)

if __name__ == "__main__":
    print("\n¡Finalizada la guía de codificación de datos categóricos y predicción de CO2!")
