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

# Resultados explicados
# La matriz de precision tiene 4 cuadrantes diferentes:
# Verdadero negativo
# Falso positivo
# Falso negativo
# Verdadero positivo
# Verdadero significa que los valores se predijeron con precision, Falso significa que hubo un error o una prediccion incorrecta.

# Metricas creadas
# La matriz proporciona metricas utiles, las diferentes metricas incluyen: exactitud, precision, sensibilidad, especificidad y la puntuacion F.

# Exactitud
# La exactitud mide la frecuencia con la que el modelo es correcto.
# La formula para calcularla es (Verdadero positivo + Verdadero negativo) / Predicciones totales
# ¿Que valor es mejor? Cercano a 0 o a 1?
accuracy = metrics.accuracy_score(actual, predicted)
print(accuracy)

# Presicion
# De los positivos previos, que porcentaje es realmente positivo?
# La formula para calcularla es Verdadero positivo / (Verdadero positivo + Falso positivo)
# La presicion no evalua los casos negativos predichos correctamente.
precision = metrics.precision_score(actual, predicted)
print(precision)

# Sensibilidad (Recuerdo)
# De todos los casos positivos que porcentaje se prevee positivo?
# La sensibilidad mide que tan bueno es el modelo para predecir resultados positivos.
# Esto significa que analiza los Verdaderos positivos y los Falsos negativos (que son positivos que se han predicho incorrectamente como negativos).
# La formula para calcularla es Verdadero positivo / (Verdadero positivo + Falso negativo)
sensitivity_recall = metrics.recall_score(actual, predicted)
print(sensitivity_recall)

# Especificidad
# ¿Que tan bueno es el modelo para predecir resultados negativos?
# La especificidad es similar a la sensibilidad, pero lo analiza desde la perspectiva de los resultados negativos.
# La formula para calcularla es Verdadero negativo / (Verdadero negativo + Falso positivo)
specificity = metrics.recall_score(actual, predicted, pos_label=0)
print(specificity)
