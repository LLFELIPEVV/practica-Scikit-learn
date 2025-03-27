"""
Evaluación de Modelos con Curvas ROC y AUC
==========================================

El área bajo la curva ROC (AUC) mide qué tan bien un modelo separa dos clases,
siendo más útil en datos desbalanceados donde la precisión no siempre refleja
el desempeño real.

Ejemplo:
  - Se simulan conjuntos de datos desbalanceados con modelos de clasificación binaria.
  - Se analizan la matriz de confusión y las curvas ROC para evaluar su desempeño.

Se utilizará Seaborn para mejorar las visualizaciones.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Configuración de estilo
sns.set_theme(style="whitegrid", font_scale=1.2)

# -----------------------------------------------------------------------------
# Datos desequilibrados
# -----------------------------------------------------------------------------
n = 10000
ratio = 0.95  # 95% de la clase mayoritaria
n_0 = int((1 - ratio) * n)
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)

# Modelo inicial sin poder de separación
y_proba = np.ones(n)
y_pred = y_proba > 0.5

# Evaluación del modelo
print(f'Accuracy Score: {accuracy_score(y, y_pred)}')
cf_mat = confusion_matrix(y, y_pred)
print('Confusion Matrix:\n', cf_mat)
print(f'Clase 0 Accuracy: {cf_mat[0][0] / n_0:.2f}')
print(f'Clase 1 Accuracy: {cf_mat[1][1] / n_1:.2f}')

# -----------------------------------------------------------------------------
# Modelo con mejor discriminación de clases
# -----------------------------------------------------------------------------
y_proba_2 = np.concatenate([
    np.random.uniform(0, 0.7, n_0),
    np.random.uniform(0.3, 1, n_1)
])
y_pred_2 = y_proba_2 > 0.5

# Evaluación del modelo mejorado
print(f'\nAccuracy Score: {accuracy_score(y, y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print('Confusion Matrix:\n', cf_mat)
print(f'Clase 0 Accuracy: {cf_mat[0][0] / n_0:.2f}')
print(f'Clase 1 Accuracy: {cf_mat[1][1] / n_1:.2f}')


# -----------------------------------------------------------------------------
# Función para graficar curvas ROC
# -----------------------------------------------------------------------------
def plot_roc_curve(true_y, y_prob, title):
    fpr, tpr, _ = roc_curve(true_y, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, label=f'AUC: {roc_auc_score(true_y, y_prob):.2f}', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()


# Curvas ROC
plot_roc_curve(y, y_proba, "Modelo 1 - ROC Curve")
plot_roc_curve(y, y_proba_2, "Modelo 2 - ROC Curve")

# -----------------------------------------------------------------------------
# Modelos con diferentes probabilidades de predicción
# -----------------------------------------------------------------------------
y = np.array([0] * n + [1] * n)

y_prob_1 = np.concatenate([
    np.random.uniform(0.25, 0.5, n//2),
    np.random.uniform(0.3, 0.7, n),
    np.random.uniform(0.5, 0.75, n//2)
])

y_prob_2 = np.concatenate([
    np.random.uniform(0, 0.4, n//2),
    np.random.uniform(0.3, 0.7, n),
    np.random.uniform(0.6, 1, n//2)
])

# Comparación de métricas
print(f'\nModelo 1 Accuracy: {accuracy_score(y, y_prob_1 > 0.5):.2f}')
print(f'Modelo 2 Accuracy: {accuracy_score(y, y_prob_2 > 0.5):.2f}')
print(f'Modelo 1 AUC Score: {roc_auc_score(y, y_prob_1):.2f}')
print(f'Modelo 2 AUC Score: {roc_auc_score(y, y_prob_2):.2f}')

# Curvas ROC para ambos modelos
plot_roc_curve(y, y_prob_1, "Modelo 1 - ROC Curve")
plot_roc_curve(y, y_prob_2, "Modelo 2 - ROC Curve")

if __name__ == "__main__":
    print("\nFinalizada la evaluación de modelos con AUC-ROC.")
