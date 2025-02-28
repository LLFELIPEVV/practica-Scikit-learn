import matplotlib.pyplot as plt

from scipy import stats

# Regresion: Este termino se utiliza cuando se intenta encontrar la relacion entre variables.
# En el machine learning y en el modelado estadistico, esta relacion se utiliza para predecir resultados de eventos futuros.

# Regresion lineal: Utiliza la relacion entre los puntos de datos para dibujar una linea recta atravez de todos ellos. Esta linea se utiliza para predecir valores futuros.

# Existe una forma matematica para encontrar esta relacion pero Python tiene metodos para encontrar esta relacion.
# Ejemplo: El eje x representa la edad y el eje y la velocidad de los datos de 13 vehiculos.
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

# Explicacion del ejemplo
# Se crean las matrices que representan los valores en x e y.
# Se ejecuta el metodo para almacenar algunos valores clave importantes de la regresion lineal.
# Se crea un funcion que utilice los valores slope y intercept, para devolver un nuevo valor. Este nuevo valor representa en que parte del eje se colocara y al valor correspodiente x.
# Se genera una nueva matriz con el resultado de la funcion.
# Se dibuja el diagrama de dispersion.
# Se dibuja la linea de regresion lineal.

# R de relacion
# Es importante saber como es la relacion entre los valores del eje x y los valores del eje y, ya que si no existe una relacion no se puede usar la regresion lineal para realizar predicciones.
# Esta relacion "el coeficiente de relacion" se llama r.
# El valor r se encuentra entre -1 y 1, donde 0 significa que no existe relacion, y 1 (y -1) significa una relacion del 100%.
# Para obtener este valor, se utiliza el metodo "linregress" del modulo "scipy" (que se importo como "stats").

# Ejemplo: Se crean las matrices que representan los valores en x e y.
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

# Predecir valores futuros
# Ahora podemos utilizar la informacion que hemos reunido para producir valores futuros.
# Ejemplo: Intentemos predecir la velocidad de un coche de 10 años.

speed = myfunc(10)
print(speed)

# Mal ajuste
# Creamos un ejemplo en el que la regresion lineal no es el mejor metodo para realizar predicciones.
# Ejemplo: Estos valores tienen una relacion muy mala.
x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20,
     26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10,
     26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
