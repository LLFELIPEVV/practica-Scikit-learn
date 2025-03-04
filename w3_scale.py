"""
Guía de Estandarización y Regresión Múltiple en Machine Learning
=================================================================

Cuando trabajamos con datos, a menudo las variables tienen diferentes rangos
y unidades de medida. Para poder compararlas y utilizarlas en un modelo de Machine Learning,
es necesario escalarlas. La estandarización es un método que transforma cada valor "x" según la fórmula:

    z = (x - u) / s

donde:
    - z es el valor estandarizado.
    - x es el valor original.
    - u es la media de los datos.
    - s es la desviación estándar.

En este ejemplo, usaremos la librería scikit-learn para:
  1. Leer datos de un archivo CSV usando Pandas.
  2. Estandarizar las variables independientes ('Weight' y 'Volume').
  3. Ajustar un modelo de regresión lineal múltiple para predecir las emisiones de CO2.
  4. Predecir nuevos valores utilizando la misma escala.
"""

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# Paso 1: Leer el archivo CSV y crear un DataFrame.
# Nota: Asegúrate de que "data.csv" esté en el mismo directorio o proporciona la ruta correcta.
df = pd.read_csv("data.csv")
print("Primeras filas del DataFrame:")
print(df.head())

# Paso 2: Seleccionar las variables independientes.
# En este ejemplo, usamos 'Weight' y 'Volume' para predecir 'CO2'.
X = df[['Weight', 'Volume']]

# Paso 3: Crear un objeto StandardScaler para estandarizar los datos.
# Esto transformará los valores de X para que tengan media 0 y desviación estándar 1.
scaler = StandardScaler()

# Ajustar el scaler a X y transformarlo.
scaledX = scaler.fit_transform(X)
print("\nDatos escalados (Weight y Volume):")
print(scaledX)

# Paso 4: Definir la variable dependiente (target), en este caso 'CO2'.
y = df['CO2']

# Paso 5: Crear y entrenar un modelo de regresión lineal múltiple usando los datos escalados.
regression_model = linear_model.LinearRegression()
regression_model.fit(scaledX, y)

# Paso 6: Para predecir valores futuros, debemos transformar la nueva observación con el mismo scaler.
# Ejemplo: Predecir las emisiones de CO2 para un vehículo con:
#    - Peso: 2300 kg
#    - Volumen: 1.3 (la unidad dependerá de tus datos; por ejemplo, litros o cm³)
new_data = [[2300, 1.3]]
scaled_new_data = scaler.transform(new_data)  # Escalar la nueva observación

# Paso 7: Realizar la predicción usando el modelo entrenado.
predicted_CO2 = regression_model.predict(scaled_new_data)
print("\nEmisiones de CO2 predichas para un vehículo con 2300 kg y 1.3 de volumen:")
print(predicted_CO2)

if __name__ == "__main__":
    print("\n¡Finalizada la guía de estandarización y regresión múltiple!")
