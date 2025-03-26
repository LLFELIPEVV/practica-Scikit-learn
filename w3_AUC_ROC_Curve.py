import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Curva AUC-ROC
# En la clasificacion existen, muchas metricas de evaluacion diferentes. La mas popular es la presicion, que mide la frecuencia con la que el modelo acierta.
# En algunos casos, se podria considerar el uso de otra metrica de evaluacion.
# Una de esas metricas es el AUC (area bajo la curva ROC). Esta curva representa la tasa de verdaderos positivos frente a la de falsos positivos en diferentes umbrales de clasificacion. Los umbrales son diferentes puntos de corte de probabilidad que separan las dos clases en la clasificacion binaria.
# Esta utiliza la probabilidad para indicar la presicion con la que un modelo separa clases.

# Datos desequilibrados
# Supongamos que tenemos un conjunto de datos desequilibrados donde la mayoria de nuestros datos tienen un solo valor.
# Podemos obtener una alta presicion del modelo prediciendo la clase mayoritaria.
n = 10000
ratio = .95
n_0 = int((1 - ratio) * n)
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)

y_proba = np.array([1] * n)
y_pred = y_proba > 0.5

print(f'accuracy score: {accuracy_score(y, y_pred)}')
cf_mat = confusion_matrix(y, y_pred)
print('Confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

# Aunque se obtuvo una presicion muy alta, el modelo no proporciono informacion sobre los datos, por lo que no es util.
# Se predice la clase 1 con una presicion del 100% mientras que la clase 0 es correcta 0%.
# A costa de la presicion, seria mejor contar con un modelo que pueda separar ligeramente las dos clases.
