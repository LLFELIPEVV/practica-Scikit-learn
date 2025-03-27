import numpy as np
import matplotlib.pyplot as plt

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

y_proba_2 = np.array(np.random.uniform(0, .7, n_0).tolist() +
                     np.random.uniform(.3, 1, n_1).tolist())
y_pred_2 = y_proba_2 > 0.5

print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print('Confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

# Para este modelo tenemos menos presicion que el primero, pero la presicion para cada clase es mas equilibrada.
# Por eso si usaramos solo la metrica de presicion para puntuar un modelo el mejor modelo seria el primero.

# En casos como este, seria preferible utilizar otra metrica de evaluacion como el AUC.


def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


plot_roc_curve(y, y_proba)
print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}')

plot_roc_curve(y, y_proba_2)
print(f'model 2 AUC score: {roc_auc_score(y, y_proba_2)}')

# Una puntuacion de AUC de 0,5 significa que el modelo no es capaz de distinguir entre las dos clases y la curva se veria como una linea pendiente de 1.
# Una puntuacion de AUC mas cercana a 1, significaria que el modelo tiene la capacidad de separar las dos clases, y la curva se acercaria a la esquina superior izquierda.

# Probabilidades
# Debido a que el AUC es una metrica que utiliza probabilidades de las predicciones de clase, nos permite saber que modelo es mejor incluso si tienen presiciones similares.
# En el siguiente ejemplo se usaran dos modelos, el primero con probabilidades menos fiables al predecir las dos clases y el segundo con probabilidades mas fiables de predecir las dos clases.
y = np.array([0] * n + [1] * n)
y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.6, 1, n//2).tolist()
)

print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1 > .5)}')
print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2 > .5)}')

print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')

plot_roc_curve(y, y_prob_1)

fpr, tpr, thresholds = roc_curve(y, y_prob_2)
plt.plot(fpr, tpr)
plt.show()
