import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# K-medias
# K-means es un metodo de aprendizaje no supervisado para agrupar puntos de datos.
# El algoritmo divide iterativamente los puntos de datos en K grupos, minimizando la varianza de cada uno.

# ¿Como funciona?
# Primero se asigna aleatoriamente cada punto a uno de los K-clusters luego se calcula el centroide de cada cluster y luego se reasigna cada punto al cluster con el centroide mas cercano y se repite este proceso hasta que las asignaciones de cluster para cada punto ya no cambien.
# La agrupacion por K-medias requiere seleccionar K, que es el numero de clusters en los que queremos agrupar los datos. El metodo del codo permite graficar la inercia (una metrica basada en la distancia) y visualizar el punto en el que comienza a disminuir linealmente. Este punto se denomina codo y constituye una buena estimacion del valor optimo de K segun nuestros datos.
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# El metodo del codo muestra que 2 es un buen valor para K, por lo que volveremos a entrenar y visualizamos el resultado.
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
