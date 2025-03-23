import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Bagging
# Metodos como los arboles de decision pueden ser propensos a sobreajustes en el conjunto de entrenamiento, lo que puede generar predicciones erroneas en nuevos datos.
# La agregacion bootstrap es un metodo de ensamblaje que intenta resolver el sobreajuste en problemas de clasificacion o regresion. Bagging busca mejorar la presicion y el rendimiento de los algoritmos de aprendizaje automatico. Para ello, toma subconjuntos aleatorios de un conjunto de datos original, con remplazo, y ajusta un clasificador o un regresor a cada subconjunto.
# Las predicciones de cada subconjunto se agregan mediante votacion mayoritaria en el caso de clasificacion y promediando para la regresion, lo que aumenta la presicion de la prediccion.

# Evaluacion de un clasificador base
# Primero revisaremos el rendimiento del clasificador base en el conjunto de datos.
# Haremos un modelo que identifique diferentes clases de vinos.
data = datasets.load_wine(as_frame=True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22)

dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print("Train data accuracy:", accuracy_score(
    y_true=y_train, y_pred=dtree.predict(X_train)))
print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))

# El clasificador base funciona bastante bien en el conjunto de datos y logra una presicion del 82% en el conjunto de pruebas con los parametros actuales.

# Creando un clasificador Bagging
# Para realizar el bagging necesitamos establecer el parametro n_estimators, este es el numero de clasificadores base que el modelo va a agregar.
# Para este conjunto de datos el numero de estimadores que se usaran es relativamente bajo, a menudo se exploran rangos mucho mas amplios. Este ajuste de hiperparametros suele realizarse mediante busqueda de cuadricula.
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

models = []
scores = []

for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    clf.fit(X_train, y_train)

    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)

plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)

plt.show()

# Resultados explicados
# Se observa un incremento en el rendimiento del modelo del 82,2% al 95,5%.
# El mejor rendimiento se observa en el rango de 10-14 estimadores.

# Otra forma de evaluacion
# Debido a que bootstrap selecciona subconjuntos aleatorios de observaciones para crear clasificadores, algunas se omiten en el proceso, por eso estas observaciones "fuera de bolsa" pueden utilizarse para evaluar el modelo.
# Se evaluara el modelo con la puntuacion out-of-bag.
oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
oob_model.fit(X_train, y_train)
print(oob_model.oob_score_)

# Generacion de arboles de decision a partir del clasificador Bagging.
clf = BaggingClassifier(n_estimators=12, random_state=22, oob_score=True)
clf.fit(X_train, y_train)
plt.figure(figsize=(30, 20))
plot_tree(clf.estimators_[0], feature_names=X.columns)
plt.show()
