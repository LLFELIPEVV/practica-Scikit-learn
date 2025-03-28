import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

# KNN es un algoritmo de aprendizaje automatico supervisado.
# K es el numero de vecinos mas cercanos que se utilizaran para la clasificacion.
# Para la clasificacion se utiliza una mayoria de votos para determinar la clase a la que debe pertenecer una nueva observacion.
# Valores mayores de K suelen ser mas robustos a los valores atipicos y producen limites de decision mas estables que valores muy pequeños.
x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

plt.scatter(x, y, c=classes)
plt.show()

# Ahora se creara el modelo con el algoritmo KNN con K=1
data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)

new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()

# Ahora con un valor mayor para K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data, classes)

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
