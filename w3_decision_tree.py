import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# Aprendizaje automatico: arbol de decisiones.
# Arbol de decision
# Es un diagrama de flujo que ayuda a tomar decisiones basadas en experiencias previas.
# Ejemplo: Una persona intenta decidir si debe ir a un espectaculo de comedia o no.
# Esta persona anteriormente registro cada vez que hubo un espectaculo de comedia en la ciudad y registro cierta informacion sobre el comediante, y tambien registro si fue o no.
df = pd.read_csv('espectaculo.csv')
print(df)

# Para crear un arbol de decisiones, todos los datos deben ser numericos.
# Debemos convertir las columnas no numericas como 'Nacionalidad' e 'Ir' en valores numericos.
# Pandas tiene un metodo map() que toma un diccionario con informacion sobre como convertir los valores.
# {'UK': 0, 'USA': 1, 'N': 2} Significa convertir los valores 'Reino Unido' a 0, 'Estados Unidos' a 1 y 'N' a 2.
# El metodo get_dummies() convierte las columnas no numericas en valores numericos.
# Ejemplo: 'Nacionalidad' tiene los valores 'UK', 'USA' y 'N'. Se convierten en 0, 1 y 2.
# Ejemplo: 'Ir' tiene los valores 'Si' y 'No'. Se convierten en 0 y 1.
d = {'Reino Unido': 0, 'EE.UU': 1, 'norte': 2}
df['Nacionalidad'] = df['Nacionalidad'].map(d)
d = {'SI': 0, 'NO': 1}
df['Ir'] = df['Ir'].map(d)
print(df)

# Luego hay que separar las columnas de caracteristicas de la columna objetivo.
# Las columnas de caracteristicas son las columnas que se usan para hacer la prediccion.
# La columna objetivo es la columna que se quiere predecir.
# Ejemplo: 'Ir' es la columna objetivo.
# Las columnas de caracteristicas son 'Nacionalidad' y 'Edad'.
features = ['Edad', 'Experiencia', 'Rango', 'Nacionalidad']

X = df[features]
y = df['Ir']

print(X)
print(y)

# Ahora podemos crear el arbol de decisiones.
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
plt.show()

"""
Explicación del Árbol de Decisión para Recomendar Comediantes
==============================================================

Este ejemplo ilustra cómo un árbol de decisión utiliza múltiples criterios (o "preguntas")
para calcular la probabilidad de que quieras ir a ver a un comediante ("SI") o no ("NO").

Cada nodo del árbol evalúa una condición y, basándose en la respuesta (True o False),
divide el conjunto de comediantes en dos ramas. En cada nodo se muestran varios valores:

1. **Condición evaluada:**
   Por ejemplo, "Rank <= 6.5". Esto significa que se compara el rango (una medida de calidad o
   preferencia) de cada comediante.
   - Si el rango es 6.5 o menor, se sigue la rama "Cierto" (True, generalmente hacia la izquierda).
   - Si es mayor que 6.5, se sigue la rama "Falso" (False, generalmente hacia la derecha).

2. **Índice Gini (gini):**
   El índice Gini mide la impureza de un nodo, es decir, qué tan mezcladas están las clases en ese
   punto. Sus valores varían de 0.0 a 0.5:
   - **0.0:** Todas las muestras en ese nodo son de la misma clase (completamente puros).
   - **0.5:** La división es perfectamente equilibrada (por ejemplo, 50% "SI" y 50% "NO").
   En nuestro ejemplo, un valor de 0.497 indica que la mezcla de respuestas es casi equilibrada.

3. **Número de muestras (samples):**
   Indica cuántos comediantes se encuentran en ese nodo del árbol.
   - Por ejemplo, "samples = 13" en el nodo raíz indica que se están considerando todos los 13 comediantes.

4. **Distribución de valores (value):**
   Se muestra como una lista, donde cada posición indica la cantidad de muestras de una determinada
   clase.
   - Por ejemplo, "value = [6, 7]" indica que, de 13 comediantes, 6 tienen la respuesta "NO" y 7 tienen "SI".

A continuación, se detalla el árbol de decisión y se explican sus ramas:

----------------------------------------------------------------------
**Nodo Raíz:**
- **Condición:** Rank <= 6.5
- **Índice Gini:** 0.497
- **Muestras:** 13
- **Valor:** [6, 7]
  *Interpretación:* De los 13 comediantes, 6 obtienen "NO" y 7 obtienen "SI" en función de su rango.

----------------------------------------------------------------------
**Rama "Cierto" (True) de la condición Rank <= 6.5:**
- **Interpretación:** Se consideran los comediantes cuyo rango es 6.5 o menor.
- **Resultados en este nodo:**
  - **Índice Gini:** 0.0 (homogeneidad total; todas las muestras tienen el mismo resultado).
  - **Muestras:** 5
  - **Valor:** [5, 0]
    *Interpretación:* De estos 5 comediantes, todos (5) tienen la respuesta "NO".

----------------------------------------------------------------------
**Rama "Falso" (False) de la condición Rank <= 6.5:**
- **Interpretación:** Se consideran los comediantes cuyo rango es mayor a 6.5.
- **Resultados en este nodo:**
  - **Muestras:** 8
  - **Valor:** [1, 7]
    *Interpretación:* De estos 8 comediantes, 1 tiene "NO" y 7 tienen "SI".

  Luego, este grupo se divide utilizando la variable "Nacionalidad":

  **Nodo: Nacionalidad <= 0.5**
  - **Interpretación:** Comediantes con un valor de nacionalidad menor o igual a 0.5 (por ejemplo,
    aquellos del Reino Unido) se agrupan en la rama "Cierto".
  - **Índice Gini:** 0.219 (baja impureza, lo que indica una buena división).
  - **Muestras:** 8
  - **Valor:** [1, 7]

  Dentro de esta rama se hacen dos divisiones adicionales según la "Edad":

  **Nodo: Edad <= 35.5**
  - **Rama "Cierto":**
    - **Interpretación:** Comediantes de 35.5 años o menos.
    - **Índice Gini:** 0.375
    - **Muestras:** 4
    - **Valor:** [1, 3]
      *Interpretación:* De estos 4, 1 obtiene "NO" y 3 obtienen "SI".

    A su vez, esta rama se divide aún más:
    - **Rama "Cierto":**
      - **Muestras:** 2
      - **Índice Gini:** 0.0
      - **Valor:** [0, 2]
        *Interpretación:* Ambos comediantes tienen "SI".
    - **Rama "Falso":**
      - **Muestras:** 2
      - **Se evalúa la variable "Experiencia": Experience <= 9.5
        - **Índice Gini:** 0.5
        - **Valor:** [1, 1]
          *Interpretación:* Hay una división equilibrada (1 "NO" y 1 "SI").
        Luego, se divide:
        - **Rama "Cierto":**
          - **Muestras:** 1
          - **Índice Gini:** 0.0
          - **Valor:** [0, 1]
            *Interpretación:* Este comediante tiene "SI".
        - **Rama "Falso":**
          - **Muestras:** 1
          - **Índice Gini:** 0.0
          - **Valor:** [1, 0]
            *Interpretación:* Este comediante tiene "NO".

  **Nodo: Edad > 35.5**
  - **Interpretación:** Comediantes con más de 35.5 años.
  - **Índice Gini:** 0.0
  - **Muestras:** 4
  - **Valor:** [0, 4]
    *Interpretación:* Todos estos 4 comediantes tienen la respuesta "SI".

----------------------------------------------------------------------
**Resumen General del Árbol:**
- El árbol de decisión comienza evaluando el rango de los comediantes.
- Se realizan divisiones sucesivas basadas en otras características (nacionalidad, edad, experiencia).
- En cada nodo, se muestran:
  - **Gini:** Un indicador de cuán "puro" es el nodo (0 = todos iguales, 0.5 = mezcla equilibrada).
  - **Samples:** La cantidad de comediantes en ese nodo.
  - **Value:** La distribución de las respuestas "NO" y "SI".

El propósito del árbol es utilizar las respuestas de las decisiones anteriores para calcular las probabilidades de que desees ver a un comediante.
Por ejemplo, si un comediante tiene un rango de 6.5 o menor, es más probable que reciba una respuesta "NO", y a partir de allí, se evalúan otras características para afinar la decisión final.

Esta estructura permite al árbol de decisión aprender de la experiencia y, a partir de múltiples divisiones, predecir de forma lógica el resultado en base a varios atributos.
"""

# Predecir valores
# Podemos usar el arbol de decision para predecir valores.
# Ejemplo: ¿Debería ir a ver un espectáculo protagonizado por un comediante estadounidense de 40 años, con 10 años de experiencia y un ranking de comedia de 7?
nuevos_datos = pd.DataFrame([[40, 10, 7, 1]], columns=features)

print(dtree.predict(nuevos_datos))

nuevos_datos = pd.DataFrame([[40, 10, 6, 1]], columns=features)

print(dtree.predict(nuevos_datos))
