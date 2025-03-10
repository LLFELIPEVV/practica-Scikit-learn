import numpy
import matplotlib.pyplot as plt

from sklearn import metrics

# Matriz de confusion
# Es una tabla que se utiliza en problemas de clasificacion para evaluar donde se cometieron errores en el modelo.
# Con esta tabla es facil ver que predicciones son erroneas.
# Las filas representan las clases reales a las que deberian haber pertenecido los resultados, mientras que las columnas representan las predicciones que hemos realizado.

# Creando una matriz de confusion
# Las matrices de confusion se pueden crear mediante predicciones realizadas a partir de una regresion logistica.
# Generar los numeros para los valores reales y previstos.
actual = numpy.random.binomial(1, 0.9, size=1000)
predicted = numpy.random.binomial(1, 0.9, size=1000)

# Usar la funcion de matriz de confusion.
confusion_matrix = metrics.confusion_matrix(actual, predicted)

# Crear la representacion mas interpretable.
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.show()
