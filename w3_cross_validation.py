from sklearn import datasets

# Validacion Cruzada
# Al ajustar los modelos se busca mejorar el rendimiento general con datos no analizados. Ajustar los hiperparametros puede mejorar considerablemente el rendimiento de los conjuntos de prueba. Sin embargo optimizar los parametros para el conjunto de prueba puede provocar fugas de informacion, lo que empeora el rendimiento del modelo con datos no analizados. Para corregir esto, se puede realizar una validacion cruzada.
X, y = datasets.load_iris(return_X_y=True)
# Exiten muchos metodos para realizar la validacion cruzada.
