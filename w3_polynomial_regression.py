"""
Guía de Regresión Polinomial y Evaluación del Ajuste (R²)
=========================================================

La regresión polinomial se utiliza cuando los datos no se ajustan bien a una línea recta.
En lugar de una recta, se utiliza una función polinomial para modelar la relación entre
las variables x e y. Esto es útil para predecir valores futuros basados en una tendencia
observada en los datos.

En este ejemplo veremos:
1. Cómo ajustar una regresión polinomial de grado 3 a un conjunto de datos.
2. Cómo visualizar el diagrama de dispersión y la curva de regresión usando Seaborn y Matplotlib.
3. Cómo calcular el coeficiente de determinación (R²) para evaluar el ajuste del modelo.
4. Cómo predecir valores futuros utilizando el modelo.
5. Un ejemplo de mal ajuste donde la regresión polinomial no captura la relación entre las variables.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Configurar Seaborn para mejorar la estética de los gráficos
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------------
# Ejemplo 1: Regresión Polinomial con Datos de Automóviles en un Peaje
# -------------------------------------------------------------------------
# Datos:
# - El eje x representa las horas del día en que pasan automóviles por un peaje.
# - El eje y representa la velocidad de los automóviles (en km/h).
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# Ajustar un modelo polinomial de grado 3 a los datos usando np.polyfit.
model_poly = np.poly1d(np.polyfit(x, y, 3))

# Crear una línea de valores x para dibujar la curva de forma suave.
x_line = np.linspace(min(x), max(x), 100)
y_line = model_poly(x_line)

# Graficar el diagrama de dispersión y la línea de regresión polinomial.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, color="dodgerblue", s=100,
                edgecolor="black", label="Datos")
plt.plot(x_line, y_line, color="red", label="Modelo Polinomial (grado 3)")
plt.title("Regresión Polinomial: Velocidad vs. Hora del Día")
plt.xlabel("Hora del día")
plt.ylabel("Velocidad (km/h)")
plt.legend()
plt.show()

# -------------------------------------------------------------------------
# Evaluación del Modelo: Coeficiente de Determinación (R²)
# -------------------------------------------------------------------------
# R² es una medida que indica qué tan bien el modelo explica la variabilidad de los datos.
# Su valor varía entre 0 (sin relación) y 1 (ajuste perfecto).
r_squared = r2_score(y, model_poly(x))
print(
    f"Coeficiente de Determinación (R²) para el modelo polinomial: {r_squared:.2f}")

# -------------------------------------------------------------------------
# Predicción de Valores Futuros
# -------------------------------------------------------------------------
# Usando el modelo, predecimos la velocidad de un automóvil que pasa por el peaje a las 17:00.
speed_at_17 = model_poly(17)
print(
    f"Velocidad estimada para un automóvil a las 17:00: {speed_at_17:.2f} km/h")

# -------------------------------------------------------------------------
# Ejemplo 2: Mal Ajuste en Regresión Polinomial
# -------------------------------------------------------------------------
# En este ejemplo, los datos no tienen una relación clara, por lo que la regresión polinomial
# no es adecuada.
x_bad = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20,
         26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y_bad = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10,
         26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

# Ajustar un modelo polinomial de grado 3 a los datos con mala relación.
model_bad = np.poly1d(np.polyfit(x_bad, y_bad, 3))
x_bad_line = np.linspace(min(x_bad), max(x_bad), 100)
y_bad_line = model_bad(x_bad_line)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_bad, y=y_bad, color="purple",
                s=100, edgecolor="black", label="Datos")
plt.plot(x_bad_line, y_bad_line, color="orange",
         label="Modelo Polinomial (grado 3)")
plt.title("Ejemplo de Mal Ajuste en Regresión Polinomial")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend()
plt.show()

# Calcular el R² para evaluar el mal ajuste.
r_squared_bad = r2_score(y_bad, model_bad(x_bad))
print(
    f"Coeficiente de Determinación (R²) para el mal ajuste: {r_squared_bad:.2f}")

if __name__ == "__main__":
    print("¡Finalizada la guía de regresión polinomial y evaluación del ajuste!")
