from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor

# La manera en la que funciona el flujo general de scikit-learn es que se comienza con unos datos, luego se pasan a un modelo, el modelo aprende de ellos y luego hace predicciones.
# data --> model --> predictions
# Siendo mas especificos los datos se dividen en dos partes, x e y.
# X corresponde a todos los datos que se usan para realizar la prediccion.
# Y corresponde a la prediccion de interes.
# El modelo tiene 2 fases. La primera cuando se crea el modelo y la segunda cuando aprende de los datos.
# Crear el modelo <-- Objeto de Python
# Aprender de los datos <-- .fit(X, y)

X, y = load_breast_cancer(return_X_y=True)
print(f"Datos: {X}")
print(f"Prediccion: {y}")
