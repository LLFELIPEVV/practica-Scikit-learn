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
