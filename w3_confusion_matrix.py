"""
Guía de Evaluación de Modelos: Matriz de Confusión y Métricas
=============================================================

La matriz de confusión es una herramienta para evaluar el desempeño de un modelo de clasificación.
Permite identificar en qué casos se han cometido errores, mostrando:
  - Verdaderos negativos (VN)
  - Falsos positivos (FP)
  - Falsos negativos (FN)
  - Verdaderos positivos (VP)

Además, a partir de la matriz se pueden calcular métricas importantes como:
  - Exactitud (Accuracy)
  - Precisión (Precision)
  - Sensibilidad o Recall (Sensibilidad)
  - Especificidad
  - Puntuación F1 (F1 Score)

En este ejemplo, se generan datos simulados de clasificación binaria y se evalúan estas métricas.
Se utiliza Seaborn para mejorar la visualización de la matriz de confusión.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

# Configurar Seaborn para gráficos con estilo
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)

# -------------------------------------------------------------------------
# Generar datos simulados
# -------------------------------------------------------------------------
# 'actual' representa los valores reales (etiquetas) y 'predicted' los valores predichos.
# Se usan distribuciones binomiales para simular datos de clasificación binaria.
actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

# -------------------------------------------------------------------------
# Crear la matriz de confusión
# -------------------------------------------------------------------------
cm = metrics.confusion_matrix(actual, predicted)

# Visualizar la matriz de confusión usando Seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[False, True], yticklabels=[False, True])
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")
plt.show()

# -------------------------------------------------------------------------
# Calcular y mostrar métricas de evaluación
# -------------------------------------------------------------------------
# Exactitud: (VP + VN) / Total de predicciones. (Valores cercanos a 1 indican buen desempeño)
accuracy = metrics.accuracy_score(actual, predicted)
print("Exactitud (Accuracy):", accuracy)

# Precisión: VP / (VP + FP). Indica el porcentaje de positivos predichos que son realmente positivos.
precision = metrics.precision_score(actual, predicted)
print("Precisión (Precision):", precision)

# Sensibilidad o Recall: VP / (VP + FN). Indica el porcentaje de casos positivos que se identificaron correctamente.
sensitivity_recall = metrics.recall_score(actual, predicted)
print("Sensibilidad (Recall):", sensitivity_recall)

# Especificidad: VN / (VN + FP). Mide qué tan bien se identifican los casos negativos.
# Se calcula utilizando la función recall_score cambiando el 'pos_label' a 0.
specificity = metrics.recall_score(actual, predicted, pos_label=0)
print("Especificidad:", specificity)

# Puntuación F1: Media armónica entre precisión y sensibilidad.
F1_score = metrics.f1_score(actual, predicted)
print("Puntuación F1:", F1_score)

# Imprimir todas las métricas en un diccionario
metrics_dict = {
    "Exactitud": accuracy,
    "Precisión": precision,
    "Sensibilidad (Recall)": sensitivity_recall,
    "Especificidad": specificity,
    "Puntuación F1": F1_score
}
print("\nResumen de Métricas:")
print(metrics_dict)

if __name__ == "__main__":
    print("\n¡Finalizada la guía de evaluación con matriz de confusión y métricas!")
