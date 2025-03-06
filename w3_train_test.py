"""
Guía de Evaluación de Modelos: Entrenamiento y Prueba (Train/Test Split)
========================================================================

Esta sección muestra cómo evaluar un modelo de regresión (en este caso, una regresión polinomial)
usando el método de entrenamiento/prueba. Esto consiste en dividir el conjunto de datos en dos partes:
  - 80% para entrenar el modelo.
  - 20% para probar el modelo.

El objetivo es medir la precisión del modelo mediante la métrica R², que varía de 0 (sin relación) a 1 (ajuste perfecto).

En este ejemplo:
  - Se simulan 100 clientes en una tienda.
  - X representa el número de minutos antes de realizar la compra.
  - y representa la cantidad de dinero gastado.
  
Utilizaremos Seaborn para mejorar la visualización de los gráficos.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# Configurar Seaborn para gráficos estéticos
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# -------------------------------------------------------------------------
# Generar datos simulados de 100 clientes
# -------------------------------------------------------------------------
np.random.seed(2)  # Fijamos la semilla para reproducibilidad

# X: Número de minutos antes de la compra, siguiendo una distribución normal (media=3, desviación=1)
X = np.random.normal(3, 1, 100)
# y: Cantidad de dinero gastado, generada con una distribución normal (media=150, desviación=40) y dividida por X
y = np.random.normal(150, 40, 100) / X

# Visualizar el conjunto de datos completo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X, y=y, s=70, color="dodgerblue", edgecolor="black")
plt.title("Diagrama de Dispersión: Minutos vs. Gasto")
plt.xlabel("Minutos antes de la compra")
plt.ylabel("Dinero gastado")
plt.show()

# -------------------------------------------------------------------------
# Dividir el conjunto de datos en Entrenamiento (80%) y Prueba (20%)
# -------------------------------------------------------------------------
# Para simplificar, se toman los primeros 80 datos como entrenamiento y los restantes como prueba.
train_x = X[:80]
train_y = y[:80]

test_x = X[80:]
test_y = y[80:]

# Visualizar el conjunto de entrenamiento
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_x, y=train_y, s=70,
                color="seagreen", edgecolor="black")
plt.title("Conjunto de Entrenamiento")
plt.xlabel("Minutos antes de la compra")
plt.ylabel("Dinero gastado")
plt.show()

# Visualizar el conjunto de prueba
plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_x, y=test_y, s=70,
                color="darkorange", edgecolor="black")
plt.title("Conjunto de Prueba")
plt.xlabel("Minutos antes de la compra")
plt.ylabel("Dinero gastado")
plt.show()

# -------------------------------------------------------------------------
# Ajustar un Modelo de Regresión Polinomial a los Datos de Entrenamiento
# -------------------------------------------------------------------------
# Dado el comportamiento observado en el diagrama de dispersión, se opta por una regresión polinomial
# de grado 4 para capturar la relación no lineal.
model_poly = np.poly1d(np.polyfit(train_x, train_y, 4))

# Crear una línea suave para visualizar el modelo (dividiendo el rango de X en 100 puntos)
x_line = np.linspace(0, 6, 100)
y_line = model_poly(x_line)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_x, y=train_y, s=70, color="seagreen",
                edgecolor="black", label="Datos de Entrenamiento")
plt.plot(x_line, y_line, color="red", linewidth=2,
         label="Modelo Polinomial (grado 4)")
plt.title("Regresión Polinomial en el Conjunto de Entrenamiento")
plt.xlabel("Minutos antes de la compra")
plt.ylabel("Dinero gastado")
plt.legend()
plt.show()

# -------------------------------------------------------------------------
# Evaluación del Modelo con R² (Coeficiente de Determinación)
# -------------------------------------------------------------------------
# Calcular R² para el conjunto de entrenamiento
r2_train = r2_score(train_y, model_poly(train_x))
print(f"R² en el conjunto de entrenamiento: {r2_train:.4f}")

# Calcular R² para el conjunto de prueba
r2_test = r2_score(test_y, model_poly(test_x))
print(f"R² en el conjunto de prueba: {r2_test:.4f}")

# -------------------------------------------------------------------------
# Predicción de Valores Futuros
# -------------------------------------------------------------------------
# Ejemplo: Predecir cuánto dinero gastará un cliente si permanece en la tienda aproximadamente 5 minutos.
predicted_value = model_poly(5)
print(
    f"Predicción: Un cliente que permanece 5 minutos gastará aproximadamente {predicted_value:.2f} dólares.")

if __name__ == "__main__":
    print("¡Finalizada la guía de entrenamiento y prueba de modelos!")
