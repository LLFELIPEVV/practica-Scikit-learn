from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

# Validacion Cruzada
# Al ajustar los modelos se busca mejorar el rendimiento general con datos no analizados. Ajustar los hiperparametros puede mejorar considerablemente el rendimiento de los conjuntos de prueba. Sin embargo optimizar los parametros para el conjunto de prueba puede provocar fugas de informacion, lo que empeora el rendimiento del modelo con datos no analizados. Para corregir esto, se puede realizar una validacion cruzada.
X, y = datasets.load_iris(return_X_y=True)
# Exiten muchos metodos para realizar la validacion cruzada.

# K-fold
# Los datos de prueba se dividen en k conjuntos de igual tamaño para validar el modelo.
# El modelo se entrena con k-1 pliegues del conjunto de entrenamiento.
# El pliegue restante se usa como conjunto de validacion para evaluar el modelo.
clf = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
