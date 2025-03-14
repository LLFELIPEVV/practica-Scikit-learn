"""
Guía de Regresión Logística para Clasificación
==============================================

La regresión logística es un método de Machine Learning utilizado para resolver problemas de clasificación,
donde el objetivo es predecir resultados categóricos (por ejemplo, benigno o maligno). A diferencia de la 
regresión lineal, que predice valores continuos, la regresión logística estima la probabilidad de que 
un evento ocurra.

En este ejemplo, usaremos el tamaño de un tumor (en centímetros) para predecir si es canceroso:
  - 0 representa un tumor benigno.
  - 1 representa un tumor canceroso.

Nota: Es importante que los datos de entrada (X) se transformen en una matriz de una columna, para que la 
función LogisticRegression() funcione correctamente.
"""

import numpy as np
from sklearn import linear_model

# Datos de ejemplo:
# X representa el tamaño del tumor en cm.
# Se convierte en una matriz de una columna usando reshape(-1, 1).
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92,
              4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)

# y representa la clasificación:
# 0 = tumor benigno y 1 = tumor canceroso.
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Crear y entrenar el modelo de regresión logística.
logr = linear_model.LogisticRegression()
logr.fit(X, y)

# -----------------------------------------------------------------------------
# Predicción:
# ¿Es canceroso un tumor de 3.46 cm?
# -----------------------------------------------------------------------------
predicted = logr.predict(np.array([3.46]).reshape(-1, 1))
print("Predicción para tumor de 3.46 cm (0 = benigno, 1 = canceroso):",
      predicted[0])

# -----------------------------------------------------------------------------
# Interpretación del Coeficiente:
# -----------------------------------------------------------------------------
# El coeficiente en regresión logística indica el cambio esperado en el logaritmo de las probabilidades
# (log-odds) de que el resultado sea 1 por cada unidad de cambio en X.
# Para interpretar este valor, se puede calcular la "razón de probabilidades" (odds) aplicando la
# función exponencial al coeficiente.
log_odds = logr.coef_
odds = np.exp(log_odds)
print("Razón de probabilidades (odds) del coeficiente:")
print(odds)
# Por ejemplo, si odds = 4, significa que por cada aumento de 1 cm en el tamaño del tumor,
# las probabilidades de que sea canceroso se multiplican por 4.

# -----------------------------------------------------------------------------
# Función para convertir log-odds en Probabilidad
# -----------------------------------------------------------------------------


def logit2prob(logr_model, x_values):
    """
    Calcula la probabilidad de que la observación sea positiva (tumor canceroso)
    usando el modelo de regresión logística.

    Parámetros:
      logr_model: El modelo de regresión logística entrenado.
      x_values: Valores de entrada (tamaño del tumor) para los cuales se desea calcular la probabilidad.

    Retorna:
      La probabilidad de que la observación sea positiva.
    """
    # Calcular el log-odds usando el coeficiente y la intersección del modelo
    log_odds = logr_model.coef_ * x_values + logr_model.intercept_
    # Convertir los log-odds en "odds"
    odds = np.exp(log_odds)
    # Calcular la probabilidad: odds / (1 + odds)
    probability = odds / (1 + odds)
    return probability


# Calcular la probabilidad para cada observación en X.
probabilities = logit2prob(logr, X)
print("\nProbabilidades de que cada tumor sea canceroso:")
print(probabilities)

"""
Resultados explicados:
- Para un tumor de 3.78 cm, la probabilidad de ser canceroso es aproximadamente 61%.
- Para un tumor de 2.44 cm, la probabilidad es aproximadamente 19%.
- Para un tumor de 2.09 cm, la probabilidad es aproximadamente 13%.

Estos ejemplos muestran cómo varían las probabilidades de que un tumor sea canceroso a medida 
que cambia su tamaño.
"""

if __name__ == "__main__":
    print("\n¡Finalizada la guía de regresión logística!")
