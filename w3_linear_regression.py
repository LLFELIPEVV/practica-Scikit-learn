"""
Guía de Regresión Lineal en Machine Learning
===========================================

La regresión lineal es una técnica estadística utilizada para encontrar la relación entre dos variables.
En Machine Learning, esta relación se usa para predecir resultados futuros basándose en datos existentes.

En este código, exploraremos:

1. **Regresión Lineal con Datos Simples**:
   - Se utilizará un conjunto de datos de la edad y la velocidad de 13 vehículos.
   - Se calculará la recta de mejor ajuste y se visualizará en un diagrama de dispersión.
   - Se evaluará la relación entre los datos usando el coeficiente de correlación (r).

2. **Predicción de Valores Futuros**:
   - Usaremos la ecuación de la regresión para predecir la velocidad de un vehículo de 10 años.

3. **Ejemplo de una Mala Relación**:
   - Se mostrará un conjunto de datos en el que la regresión lineal no es adecuada debido a una relación débil.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

# Configuración de Seaborn para mejorar los gráficos
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------------
# Ejemplo 1: Regresión Lineal con Datos Simples
# -------------------------------------------------------------------------
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]  # Edad de los coches
y = [99, 86, 87, 88, 111, 86, 103, 87, 94,
     78, 77, 85, 86]  # Velocidad de los coches

# Cálculo de la regresión lineal
slope, intercept, r, p, std_err = stats.linregress(x, y)


# Función que usa la ecuación de la recta de regresión
def predict(x):
    return slope * x + intercept


# Generar valores predichos para la línea de regresión
y_pred = list(map(predict, x))

# Graficar los datos y la línea de regresión
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, s=100, color="dodgerblue",
                edgecolor="black", label="Datos Reales")
plt.plot(x, y_pred, color="red", label="Regresión Lineal")
plt.title("Regresión Lineal: Edad vs. Velocidad de Coches")
plt.xlabel("Edad del coche (años)")
plt.ylabel("Velocidad del coche (km/h)")
plt.legend()
plt.show()

# Evaluar la relación entre los datos
print(f"Coeficiente de correlación (r): {r:.2f}")

# -------------------------------------------------------------------------
# Ejemplo 2: Predicción de Valores Futuros
# -------------------------------------------------------------------------
# Predecir la velocidad de un coche con 10 años
speed_pred = predict(10)
print(f"Velocidad estimada para un coche de 10 años: {speed_pred:.2f} km/h")

# -------------------------------------------------------------------------
# Ejemplo 3: Un Caso de Mala Relación
# -------------------------------------------------------------------------
x_bad = [89, 43, 36, 36, 95, 10, 66, 34, 38,
         20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y_bad = [21, 46, 3, 35, 67, 95, 53, 72, 58,
         10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

# Cálculo de la regresión lineal en datos sin una relación clara
slope_bad, intercept_bad, r_bad, p_bad, std_err_bad = stats.linregress(
    x_bad, y_bad)

# Generar valores predichos
y_pred_bad = list(map(lambda x: slope_bad * x + intercept_bad, x_bad))

# Graficar los datos sin relación clara
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_bad, y=y_bad, s=100, color="orange",
                edgecolor="black", label="Datos Reales")
plt.plot(x_bad, y_pred_bad, color="red", label="Regresión Lineal")
plt.title("Ejemplo de Mala Relación en Regresión Lineal")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend()
plt.show()

# Evaluar la relación entre los datos
print(f"Coeficiente de correlación en el caso malo (r): {r_bad:.2f}")

if __name__ == "__main__":
    print("¡Finalizada la guía de regresión lineal!")
