import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Agrupamiento jerarquico
# La agrupacion jerarquica es un metodo de apredizaje no supervisado para agrupar puntos de datos.
# El algoritmo crea grupos midiendo las diferencias entre los datos.
# El aprendizaje no supervisado significa que no es necesario entrenar un modelo y no necesitamos una variable 'objetivo'.
# Este metodo se puede utilizar con cualquier dato para visualizar e interpretar la relacion entre puntos de datos individuales.

# ¿Como funciona?
# Utilizaremos el agrupamiento aglomerativo, un tipo de agrupamiento que toma inicialmente cada punto como su propio grupo, luego une los mas cercanos para ir creando grupos mas grandes y asi sucesivamente hasta tener un solo grupo
# La agrupacion jerarquica requiere decidamos tanto el metodo de distancia como el de vinculacion.
# En este ejemplo se usara la distancia euclidiana y el metodo de vinculacion de ward, que intenta minimizar la varianza entre grupos.
# Ejemplo: calcular el vínculo de barrio utilizando la distancia euclidiana y lo visualizamos utilizando un dendrograma
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linked_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linked_data)

plt.show()

# Ahora con scikit-learn
hierarchical_cluster = AgglomerativeClustering(
    n_clusters=2, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)
plt.show()

# Ejemplo explicado
# Se crean dos matrices que se asemejen a dos variables en un conjunto de datos. En este ejemplo se usaron 2 variables pero este metodo sirve para cualquier cantidad de variables.
# Luego se convierten los datos en un conjunto de puntos.
# Luego se calcula la relacion entre todos los diferentes puntos.
# Por ultimo se grafica en un dendrograma. Este grafico muestra la jerarquia de los grupos desde la base.
