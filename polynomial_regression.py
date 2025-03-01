import numpy as np
import matplotlib.pyplot as plt

# Si los puntos de datos no se ajustan auna regresion lineal, podria ser ideal para la regresion polinomial.
# La regresion polinomial al igual que la lineal, utiliza la relacion entre x e y para encontrar la mejor manera de dibujar una linea a traves de los puntos de datos.
# Python cuenta con metodos para encontrar una relacion entre puntos de datos y para dibujar una linea de regresion polinomial.
# Ejemplo: Se registraron 18 automoviles cuando pasaban por un peaje determinado.
# El eje x representa las horas del dia, y el eje y representa la velocidad.
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

mymodel = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# Ejemplo explicado:
# Creamos las matrices que representan los valores de los ejes x e y.
# NumPy tiene un método que nos permite hacer un modelo polinomial.
# Luego especifica cómo se mostrará la línea, comenzamos en la posición 1 y terminamos en la posición 22, el 100 significa la cantidad de divisiones que puede tener la linea.
# Dibuje el diagrama de dispersión original.
# Dibuje la línea de regresión polinomial.
