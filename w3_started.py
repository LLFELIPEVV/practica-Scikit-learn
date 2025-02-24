"""
Guía Básica de Machine Learning
================================

Esta guía está diseñada para explicar conceptos fundamentales de Machine Learning
de forma sencilla, incluso para alguien sin experiencia en programación.

Cada sección incluye definiciones, ejemplos y fragmentos de código para facilitar
la comprensión de los siguientes conceptos:

1. Conjunto de Datos
2. Aprendizaje Automático (Machine Learning)
3. Tipos de Datos:
   a. Datos Numéricos (Discretos y Continuos)
   b. Datos Categóricos
   c. Datos Ordinales
"""

# -----------------------------------------------------------------------------
# 1. Conjunto de Datos (Data Set)
# -----------------------------------------------------------------------------
# Un conjunto de datos es una colección organizada de información.
# Por ejemplo, imagina una hoja de cálculo en la que cada fila representa a una persona
# y cada columna contiene información (edad, altura, género, etc.). Esa hoja es un
# conjunto de datos.

# Ejemplo en Python: Un pequeño conjunto de datos de personas.
data_set = [
    {"nombre": "Ana", "edad": 28, "altura": 1.65, "género": "Femenino"},
    {"nombre": "Luis", "edad": 32, "altura": 1.70, "género": "Masculino"},
    {"nombre": "María", "edad": 25, "altura": 1.60, "género": "Femenino"}
]

print("Conjunto de Datos:")
for registro in data_set:
    print(registro)

# -----------------------------------------------------------------------------
# 2. Aprendizaje Automático (Machine Learning)
# -----------------------------------------------------------------------------
# El aprendizaje automático es una rama de la inteligencia artificial que utiliza
# algoritmos para analizar datos, identificar patrones y hacer predicciones o
# tomar decisiones basadas en esos datos.
#
# Ejemplo: Predecir el precio de una casa según su tamaño.


def predecir_precio(tamaño_m2):
    # Supongamos que cada metro cuadrado cuesta 1000 unidades monetarias.
    return tamaño_m2 * 1000


tamaño_casa = 120  # en metros cuadrados
precio_casa = predecir_precio(tamaño_casa)
print("\nPrecio de una casa de 120 m²:", precio_casa)

# -----------------------------------------------------------------------------
# 3. Tipos de Datos
# -----------------------------------------------------------------------------

# a) Datos Numéricos
# ------------------
# Son datos representados por números y se dividen en:
#
# - Datos Discretos:
#   Son valores contados que solo pueden ser números enteros.
#   Ejemplo: El número de estudiantes en una clase (20, 21, 22, ...).
#
# - Datos Continuos:
#   Son valores medidos que pueden tomar cualquier número dentro de un rango.
#   Ejemplo: La temperatura de un día (22.5°C, 22.6°C, 22.7°C, ...).

numero_estudiantes = 25       # Ejemplo de dato discreto
temperatura_actual = 22.5       # Ejemplo de dato continuo

print("\nNúmero de estudiantes (discreto):", numero_estudiantes)
print("Temperatura (continua):", temperatura_actual)

# b) Datos Categóricos
# ---------------------
# Representan categorías o grupos y no tienen un orden numérico.
# Ejemplo: Colores (rojo, verde, azul) o tipos de fruta (manzana, naranja, banana).
#
# No se pueden comparar en términos de mayor o menor, ya que son etiquetas.

color_favorito = "rojo"
tipo_fruta = "manzana"

print("\nEjemplos de datos categóricos:")
print("Color favorito:", color_favorito)
print("Tipo de fruta:", tipo_fruta)

# c) Datos Ordinales
# ------------------
# Son datos categóricos que sí tienen un orden o jerarquía.
# Ejemplo: Calificaciones escolares (A, B, C, D, F) o niveles de satisfacción
# (muy insatisfecho, insatisfecho, neutral, satisfecho, muy satisfecho).
#
# Aquí es posible decir que "muy satisfecho" es superior a "satisfecho", etc.

nivel_satisfaccion = "muy satisfecho"

print("\nEjemplo de dato ordinal:")
print("Nivel de satisfacción:", nivel_satisfaccion)

# -----------------------------------------------------------------------------
# Resumen
# -----------------------------------------------------------------------------
# - Un conjunto de datos es como una tabla donde cada fila es un ejemplo y cada columna
#   representa una característica.
#
# - El aprendizaje automático utiliza algoritmos para aprender de los datos y hacer
#   predicciones o tomar decisiones (ejemplo: predecir precios, clasificar correos, etc.).
#
# - Tipos de datos:
#   * Datos Numéricos: Pueden ser discretos (contables) o continuos (medidos).
#   * Datos Categóricos: Etiquetas sin un orden inherente.
#   * Datos Ordinales: Etiquetas con un orden definido (por ejemplo, calificaciones).
#
# Con estos conceptos básicos, tienes una buena base para comprender cómo se estructuran
# los datos y por qué son esenciales en los modelos de Machine Learning.
#
# ¡Esta guía te servirá para comenzar tu camino en el aprendizaje automático!

if __name__ == "__main__":
    print("\n¡Bienvenido a la Guía Básica de Machine Learning!")
