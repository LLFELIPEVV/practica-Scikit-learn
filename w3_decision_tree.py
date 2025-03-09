"""
Guía de Árbol de Decisión para Recomendar Espectáculos de Comedia
=================================================================

Un árbol de decisión es un diagrama de flujo que ayuda a tomar decisiones basadas en datos históricos.
En este ejemplo, se utiliza un archivo CSV llamado "espectaculo.csv" que contiene registros de espectáculos
de comedia. La información registrada incluye datos como:
  - Edad
  - Experiencia
  - Rango (calificación del comediante)
  - Nacionalidad
  - Ir (indica si la persona asistió al espectáculo, con valores "SI" o "NO")

El árbol de decisión se construye para predecir si se debería ir a ver a un comediante
basado en estas características.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Configurar Seaborn para mejorar la apariencia de los gráficos
sns.set_theme(style="whitegrid", font_scale=1.2)

# -------------------------------------------------------------------------
# 1. Leer el archivo CSV y mostrar los datos originales
# -------------------------------------------------------------------------
df = pd.read_csv('espectaculo.csv')
print("Datos originales:")
print(df)

# -------------------------------------------------------------------------
# 2. Convertir las columnas no numéricas en valores numéricos
# -------------------------------------------------------------------------
# Para construir el árbol de decisión, todos los datos deben ser numéricos.
# Convertimos la columna 'Nacionalidad' usando un diccionario:
nacionalidad_map = {'Reino Unido': 0, 'EE.UU': 1, 'norte': 2}
df['Nacionalidad'] = df['Nacionalidad'].map(nacionalidad_map)

# Convertimos la columna 'Ir' (respuesta) en números:
ir_map = {'SI': 0, 'NO': 1}
df['Ir'] = df['Ir'].map(ir_map)

print("\nDatos después de la conversión:")
print(df)

# -------------------------------------------------------------------------
# 3. Separar las características y la variable objetivo
# -------------------------------------------------------------------------
# Las "características" (features) son los datos que usamos para predecir.
# En este caso, se usarán 'Edad', 'Experiencia', 'Rango' y 'Nacionalidad'.
# La "variable objetivo" (target) es 'Ir', que indica si la persona asistió o no.
features = ['Edad', 'Experiencia', 'Rango', 'Nacionalidad']
X = df[features]
y = df['Ir']

print("\nCaracterísticas (X):")
print(X)
print("\nVariable objetivo (y):")
print(y)

# -------------------------------------------------------------------------
# 4. Crear y entrenar el Árbol de Decisión
# -------------------------------------------------------------------------
# Se crea un objeto DecisionTreeClassifier y se entrena (fit) con X e y.
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# -------------------------------------------------------------------------
# 5. Visualizar el Árbol de Decisión
# -------------------------------------------------------------------------
# Usamos tree.plot_tree() de scikit-learn para dibujar el árbol.
plt.figure(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=features,
               filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión para Recomendar Espectáculos de Comedia")
plt.show()

"""
Explicación del Árbol de Decisión:
----------------------------------
El árbol se utiliza para predecir la probabilidad de que desees ver a un comediante basado en varios criterios.
Cada nodo del árbol evalúa una condición (por ejemplo, "Rango <= 6.5") y divide los datos en ramas:

- **Condición evaluada:** Por ejemplo, "Rango <= 6.5" compara el valor de 'Rango'.
    - Si la condición es verdadera, se sigue una rama (por lo general, a la izquierda).
    - Si es falsa, se sigue la otra rama (a la derecha).

- **Índice Gini:** Mide la impureza del nodo (0 = puro, 0.5 = mezcla).
- **Samples:** Número de muestras (comediantes) en ese nodo.
- **Value:** Distribución de las clases en ese nodo (por ejemplo, [6, 7] significa 6 "NO" y 7 "SI").

La estructura del árbol se va refinando en cada nodo, ayudándote a decidir si deberías ir a ver a un comediante.
"""

# -------------------------------------------------------------------------
# 6. Predecir con el Árbol de Decisión
# -------------------------------------------------------------------------
# Ejemplo: ¿Deberías ir a ver un espectáculo protagonizado por un comediante con las siguientes características?
# - Edad: 40 años
# - Experiencia: 10 años
# - Rango: 7 (calificación de comedia)
# - Nacionalidad: 1 (EE.UU, según nuestro mapeo)
nuevos_datos = pd.DataFrame([[40, 10, 7, 1]], columns=features)
prediccion = dtree.predict(nuevos_datos)
print("\nPredicción para los datos", nuevos_datos.to_dict(
    orient='records')[0], ":", "Ir" if prediccion[0] == 0 else "No Ir")

# Otro ejemplo: Cambiando el rango a 6
nuevos_datos2 = pd.DataFrame([[40, 10, 6, 1]], columns=features)
prediccion2 = dtree.predict(nuevos_datos2)
print("Predicción para los datos", nuevos_datos2.to_dict(
    orient='records')[0], ":", "Ir" if prediccion2[0] == 0 else "No Ir")

if __name__ == "__main__":
    print("\n¡Finalizada la guía de árboles de decisión!")
