"""
Guía de Validación Cruzada para Modelos de Clasificación
=======================================================

La validación cruzada es una técnica para evaluar el rendimiento de un modelo en datos no vistos, evitando sobreajustes.

Algunos métodos incluyen:
  1. K-Fold: Divide los datos en k subconjuntos para entrenar y validar iterativamente.
  2. Stratified K-Fold: Similar a K-Fold, pero preserva la proporción de clases.
  3. Leave-One-Out (LOO): Usa una observación como validación y el resto como entrenamiento.
  4. Leave-P-Out (LPO): Similar a LOO, pero permite elegir más de una observación para validación.
  5. Shuffle Split: Divide los datos aleatoriamente en entrenamiento y validación varias veces.

Se utilizará el dataset Iris y Seaborn para mejorar las visualizaciones.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, cross_val_score
)

# Configurar Seaborn
sns.set_theme(style="whitegrid", font_scale=1.2)

# -----------------------------------------------------------------------------
# Cargar el Dataset
# -----------------------------------------------------------------------------
X, y = datasets.load_iris(return_X_y=True)

# Crear el clasificador base
clf = DecisionTreeClassifier(random_state=42)

# -----------------------------------------------------------------------------
# 1. K-Fold Cross Validation
# -----------------------------------------------------------------------------
k_folds = KFold(n_splits=5)
kfold_scores = cross_val_score(clf, X, y, cv=k_folds)

print("\nK-Fold Cross Validation Scores:", kfold_scores)
print("Average CV Score:", kfold_scores.mean())

# -----------------------------------------------------------------------------
# 2. Stratified K-Fold Cross Validation
# -----------------------------------------------------------------------------
sk_fold = StratifiedKFold(n_splits=5)
stratified_scores = cross_val_score(clf, X, y, cv=sk_fold)

print("\nStratified K-Fold Scores:", stratified_scores)
print("Average CV Score:", stratified_scores.mean())

# -----------------------------------------------------------------------------
# 3. Leave-One-Out (LOO)
# -----------------------------------------------------------------------------
loo = LeaveOneOut()
loo_scores = cross_val_score(clf, X, y, cv=loo)

print("\nLeave-One-Out Cross Validation Scores (first 10 shown):",
      loo_scores[:10])
print("Average CV Score:", loo_scores.mean())

# -----------------------------------------------------------------------------
# 4. Leave-P-Out (LPO) con p=2
# -----------------------------------------------------------------------------
lpo = LeavePOut(p=2)
lpo_scores = cross_val_score(clf, X, y, cv=lpo)

print("\nLeave-P-Out Cross Validation Scores (first 10 shown):",
      lpo_scores[:10])
print("Average CV Score:", lpo_scores.mean())

# -----------------------------------------------------------------------------
# 5. Shuffle Split
# -----------------------------------------------------------------------------
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
ss_scores = cross_val_score(clf, X, y, cv=ss)

print("\nShuffle Split Cross Validation Scores:", ss_scores)
print("Average CV Score:", ss_scores.mean())

# -----------------------------------------------------------------------------
# Visualización de los Resultados
# -----------------------------------------------------------------------------
methods = ["K-Fold", "Stratified K-Fold", "LOO", "LPO (p=2)", "Shuffle Split"]
scores_means = [
    kfold_scores.mean(),
    stratified_scores.mean(),
    loo_scores.mean(),
    lpo_scores.mean(),
    ss_scores.mean(),
]

plt.figure(figsize=(10, 6))
sns.barplot(x=methods, y=scores_means, palette="coolwarm")
plt.ylim(0.8, 1)
plt.ylabel("Precisión Promedio")
plt.title("Comparación de Técnicas de Validación Cruzada")
plt.show()

if __name__ == "__main__":
    print("\n¡Finalizada la guía de validación cruzada!")
