# Bagging
# Metodos como los arboles de decision pueden ser propensos a sobreajustes en el conjunto de entrenamiento, lo que puede generar predicciones erroneas en nuevos datos.
# La agregacion bootstrap es un metodo de ensamblaje que intenta resolver el sobreajuste en problemas de clasificacion o regresion. Bagging busca mejorar la presicion y el rendimiento de los algoritmos de aprendizaje automatico. Para ello, toma subconjuntos aleatorios de un conjunto de datos original, con remplazo, y ajusta un clasificador o un regresor a cada subconjunto.
# Las predicciones de cada subconjunto se agregan mediante votacion mayoritaria en el caso de clasificacion y promediando para la regresion, lo que aumenta la presicion de la prediccion.
