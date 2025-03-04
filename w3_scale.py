# Esto se realiza cuando los datos tienen valores diferentes e incluso unidades de medida diferente.
# El escalamiento permite obtener nuevos valores que sean mas faciles de comparar.
# Existen varios metodos para escalar datos, el que usaremos se llama estandarizacion.
# El metodo de estandarizacion utiliza esta formula:
# z = (x - u) / s
# Donde:
# z es el nuevo valor estandarizado
# x es el valor original
# u es la media de los datos
# s es la desviacion estandar de los datos
# Para estandarizar los datos, usaremos la libreria sklearn
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

# Creamos un array con los datos
df = pd.read_csv("data.csv")

X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)
print(scaledX)
