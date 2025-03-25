from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, cross_val_score

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

# Stratified K-Fold
# En casos de desequilibrio de clases, necesitamos una forma de explicarlo tanto en los conjuntos de entrenamiento como en los de validacion. Para ello podemos estratificar las clases objetivo, lo que significa que ambos conjuntos tendran la misma proporcion de todas las clases.
sk_fold = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=sk_fold)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# El numero de pliegues es el mismo pero el CV promedio aumenta a partir del pliegue k basico cuando se asegura que haya clases estratificadas.

# Leave-One-Out (LOO)
# En lugar de seleccionar el numero de divisiones, se utiliza una observacion para validar y n-1 observaciones para entrenar. Este metodo es una tecnica exhaustiva.
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
# El numero de puntuaciones de validacion cruzada realizadas es igual al numero de observaciones en el conjunto de datos.
# En este caso hay 150 observaciones en el conjunto de iris.
# La puntuacion media de CV es del 94%

# Leave-P-Out (LPO)
# Es una diferencia matizada con respecto a LOO, en la que podemos seleccionar la cantidad de p que utilizaremos en nuestro conjunto de validacion.
lpo = LeavePOut(p=2)
scores = cross_val_score(clf, X, y, cv=lpo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
# Se calculan muchos mas puntajes que LOO, incluso con p=2 aunque logra aproximadamente el mismo puntaje CV promedio.

# Shuffle Split
# A diferencia de K-Fold, Shuffle Split excluye el porcentaje de los datos que no se utilizara en los conjuntos de entrenamiento ni de validacion. Para ello debemos determinar el tamaño del entrenamiento y de la prueba, asi como el numero de divisiones.
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores = cross_val_score(clf, X, y, cv=ss)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
