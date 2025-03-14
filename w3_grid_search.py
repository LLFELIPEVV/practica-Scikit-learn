# Busqueda de Cuadricula
# La mayoria de los modelos de aprendizaje automatico contienen parametros que pueden ajustarse para variar su aprendizaje.
# Por ejemplo el modelo de regresion logistica de sklearn tiene un parametro C que controla la regularizacion, lo cual afecta la complejidad del modelo.

# ¿Como funciona?
# Un metodo consiste en probar diferentes valores y combinaciones y seleccionar el que ofrezca la mayor puntuacion.
# Esta tecnica se conoce como busqueda en cuadricula.
# Si tuvieramos que seleccionar los valores de dos o mas parametros, evaluariamos todas la combinaciones de los conjuntos de valores, formando asi una cuadricula de valores.
# Valores altos de C le indican al modelo que los datos de entrenamiento se asemejan a la informacion del mundo real y le otorgan mayor importancia a estos. Valores de C tienen el efecto contrario.
