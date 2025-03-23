"""
Guía de Ensamblaje con Bagging (Bootstrap Aggregation) para Clasificación
========================================================================

El bagging es una técnica de ensamblaje que mejora el desempeño de modelos
como los árboles de decisión, reduciendo el sobreajuste. La idea es entrenar
varios clasificadores base en subconjuntos aleatorios (con reemplazo) del conjunto
de datos y luego combinar sus predicciones mediante votación mayoritaria.

En este ejemplo utilizaremos el dataset de vinos para:
  1. Evaluar un clasificador base (árbol de decisión) sin bagging.
  2. Aplicar bagging variando el número de estimadores (clasificadores base)
     y analizar cómo mejora la precisión.
  3. Evaluar el rendimiento mediante la puntuación "out-of-bag" (OOB).
  4. Visualizar uno de los árboles de decisión generados por el clasificador Bagging.

Se utilizará Seaborn para mejorar la apariencia de los gráficos.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Configurar Seaborn para gráficos con buen estilo
sns.set_theme(style="whitegrid", font_scale=1.2)

# -----------------------------------------------------------------------------
# 1. Evaluación del Clasificador Base (Árbol de Decisión) sin Bagging
# -----------------------------------------------------------------------------
# Cargamos el dataset de vinos
data = datasets.load_wine(as_frame=True)
X = data.data       # Características
y = data.target     # Etiquetas (clases de vino)

# Dividir los datos en conjuntos de entrenamiento y prueba (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22)

# Crear y entrenar un árbol de decisión
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)

# Realizar predicciones y evaluar la precisión
y_pred = dtree.predict(X_test)
accuracy_base_train = metrics.accuracy_score(y_train, dtree.predict(X_train))
accuracy_base_test = metrics.accuracy_score(y_test, y_pred)
print("Precisión del árbol de decisión (entrenamiento):", accuracy_base_train)
print("Precisión del árbol de decisión (prueba):", accuracy_base_test)
# Con la configuración predeterminada, el clasificador base tiene una precisión de alrededor del 82%.

# -----------------------------------------------------------------------------
# 2. Búsqueda del Mejor Número de Estimadores para Bagging
# -----------------------------------------------------------------------------
# Se probarán diferentes valores para n_estimators (número de clasificadores base)
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
models = []
scores = []

for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    clf.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, clf.predict(X_test))
    models.append(clf)
    scores.append(score)

# Visualizar la relación entre el número de estimadores y la precisión usando Seaborn
plt.figure(figsize=(9, 6))
sns.lineplot(x=estimator_range, y=scores, marker="o", color="mediumseagreen")
plt.title("Método del Codo para Bagging")
plt.xlabel("Número de Estimadores (n_estimators)", fontsize=18)
plt.ylabel("Precisión del Modelo", fontsize=18)
plt.xticks(estimator_range, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

print("Mejores resultados se observan en el rango de 10 a 14 estimadores.")

# -----------------------------------------------------------------------------
# 3. Evaluación con Out-of-Bag (OOB)
# -----------------------------------------------------------------------------
# El OOB evalúa el modelo en las observaciones no seleccionadas en cada muestra bootstrap.
oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
oob_model.fit(X_train, y_train)
print("Puntuación OOB:", oob_model.oob_score_)

# -----------------------------------------------------------------------------
# 4. Visualización de un Árbol de Decisión del Ensamble Bagging
# -----------------------------------------------------------------------------
# Extraemos uno de los árboles entrenados dentro del clasificador Bagging.
bagging_clf = BaggingClassifier(
    n_estimators=12, random_state=22, oob_score=True)
bagging_clf.fit(X_train, y_train)

# Visualizar el primer árbol del ensamble
plt.figure(figsize=(30, 20))
plot_tree(bagging_clf.estimators_[
          0], feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión Extraído del Clasificador Bagging", fontsize=20)
plt.show()

# -----------------------------------------------------------------------------
# 5. Resumen de Resultados
# -----------------------------------------------------------------------------
results = {
    "Precisión del Árbol Base (Entrenamiento)": accuracy_base_train,
    "Precisión del Árbol Base (Prueba)": accuracy_base_test,
    "Mejores Puntuaciones con Bagging": scores,
    "Puntuación OOB": oob_model.oob_score_
}
print("\nResumen de Métricas y Evaluaciones:")
print(results)

if __name__ == "__main__":
    print("\n¡Finalizada la guía de Bagging con clasificación!")
