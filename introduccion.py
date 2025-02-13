# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt  # Para graficar los resultados
# Para cargar el conjunto de datos
from sklearn.datasets import load_breast_cancer
# Para usar el modelo de regresión KNN
from sklearn.neighbors import KNeighborsRegressor

# En scikit-learn, el flujo de trabajo general es el siguiente:
# 1. Cargar o generar los datos.
# 2. Dividir los datos en dos partes: X (características) e y (etiquetas o valores a predecir).
# 3. Crear un modelo de aprendizaje automático.
# 4. Entrenar el modelo con los datos (X, y).
# 5. Usar el modelo entrenado para hacer predicciones.

# Cargamos el conjunto de datos de cáncer de mama
# `X` contiene las características (datos que se usan para hacer predicciones).
# `y` contiene las etiquetas (lo que queremos predecir).
X, y = load_breast_cancer(return_X_y=True)

# Mostramos los datos cargados
# Muestra las características (datos de entrada)
print(f"Características (X): {X}")
# Muestra las etiquetas (lo que queremos predecir)
print(f"Etiquetas (y): {y}")

# Creamos el modelo de regresión K-Nearest Neighbors (KNN)
# KNN es un algoritmo que predice el valor de un punto basándose en los valores de los puntos más cercanos.
mod = KNeighborsRegressor()

# Entrenamos el modelo con los datos
# El método `fit` hace que el modelo "aprenda" de los datos (X, y).
mod.fit(X, y)

# Usamos el modelo entrenado para hacer predicciones sobre los mismos datos (X)
# `pred` contendrá las predicciones que el modelo hace para cada valor de X.
pred = mod.predict(X)

# Graficamos los resultados
# Usamos un gráfico de dispersión para comparar las predicciones (`pred`) con los valores reales (`y`).
plt.scatter(pred, y)  # Grafica las predicciones vs los valores reales
plt.xlabel("Predicciones")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
plt.title("Predicciones vs Valores reales")  # Título del gráfico
plt.show()  # Muestra el gráfico
