from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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
