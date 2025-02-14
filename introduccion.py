# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt  # Para crear gráficos
import seaborn as sns  # Para hacer gráficos más bonitos y atractivos
import pandas as pd  # Para trabajar con datos en forma de tablas (como Excel)

# Importamos herramientas de scikit-learn (una biblioteca para machine learning)
# Para cargar un conjunto de datos de diabetes
from sklearn.datasets import load_diabetes
# Para usar un modelo de predicción llamado KNN
from sklearn.neighbors import KNeighborsRegressor
# Para escalar los datos (ajustar su escala)
from sklearn.preprocessing import StandardScaler
# Para crear un flujo de trabajo que combine varias etapas
from sklearn.pipeline import Pipeline
# Para buscar los mejores ajustes del modelo
from sklearn.model_selection import GridSearchCV

# En machine learning, el proceso general es el siguiente:
# 1. Cargar los datos.
# 2. Dividir los datos en dos partes: X (características) e y (lo que queremos predecir).
# 3. Crear un modelo que pueda aprender de los datos.
# 4. Entrenar el modelo con los datos.
# 5. Usar el modelo para hacer predicciones.

# Cargamos el conjunto de datos de diabetes
# `X` contiene las características (datos que usamos para hacer predicciones).
# `y` contiene las etiquetas (lo que queremos predecir, en este caso, valores relacionados con la diabetes).
X, y = load_diabetes(return_X_y=True)

# Mostramos los datos cargados
# Muestra las características (datos de entrada)
print(f"Características (X): {X}")
# Muestra las etiquetas (lo que queremos predecir)
print(f"Etiquetas (y): {y}")

# Configuramos el estilo de los gráficos con Seaborn
# Usamos un fondo con cuadrícula para que los gráficos se vean mejor
sns.set_theme(style="whitegrid")

# --- Primer caso: Sin escalar los datos ---
# Creamos un modelo de regresión K-Nearest Neighbors (KNN)
# KNN es un algoritmo que predice el valor de un punto basándose en los valores de los puntos más cercanos.
mod = KNeighborsRegressor()

# Entrenamos el modelo con los datos
# El método `fit` hace que el modelo "aprenda" de los datos (X, y).
mod.fit(X, y)

# Usamos el modelo entrenado para hacer predicciones sobre los mismos datos (X)
# `pred_sin_escalado` contendrá las predicciones que el modelo hace para cada valor de X.
pred_sin_escalado = mod.predict(X)

# Graficamos los resultados sin escalar los datos
plt.figure(figsize=(10, 5))  # Definimos el tamaño del gráfico
# Creamos el primer gráfico (1 fila, 2 columnas, posición 1)
plt.subplot(1, 2, 1)

# Usamos un gráfico de dispersión para comparar las predicciones con los valores reales
sns.scatterplot(x=pred_sin_escalado, y=y, alpha=0.6,
                color='blue')  # Puntos azules con transparencia
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia (y = x)
plt.xlabel("Predicciones (sin escalado)")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
# Título del gráfico
plt.title("Predicciones vs Valores reales (sin escalado)")

# --- Segundo caso: Escalando los datos ---
# Es mejor escalar los datos antes de pasarlos al modelo.
# Escalar significa ajustar los datos para que tengan una media de 0 y una desviación estándar de 1.
# Creamos una tubería (pipeline) que primero escala los datos y luego aplica el modelo KNN.
pipe = Pipeline([
    ("scale", StandardScaler()),  # Escala los datos
    ("model", KNeighborsRegressor(n_neighbors=1))  # Aplica el modelo KNN
])

# Entrenamos la tubería con los datos
pipe.fit(X, y)

# Hacemos predicciones sobre los mismos datos (X)
pred_con_escalado = pipe.predict(X)

# Graficamos los resultados con los datos escalados
# Creamos el segundo gráfico (1 fila, 2 columnas, posición 2)
plt.subplot(1, 2, 2)
sns.scatterplot(x=pred_con_escalado, y=y, alpha=0.6,
                color='green')  # Puntos verdes con transparencia
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia (y = x)
plt.xlabel("Predicciones (con escalado)")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
# Título del gráfico
plt.title("Predicciones vs Valores reales (con escalado)")

# --- Tercer caso: Buscando los mejores ajustes del modelo ---
# Para mejorar el modelo, podemos buscar los mejores hiperparámetros (ajustes del modelo).
# Usamos una técnica llamada GridSearchCV, que prueba diferentes combinaciones de parámetros.
# Además, divide los datos en partes para entrenar y validar el modelo varias veces (esto se llama validación cruzada).
# Muestra los parámetros que podemos ajustar
print(f"Parámetros disponibles: {pipe.get_params()}")

# Creamos un objeto GridSearchCV para buscar los mejores valores de `n_neighbors` (número de vecinos en KNN)
mod = GridSearchCV(
    estimator=pipe,  # Usamos la tubería que creamos antes
    # Probamos valores de 1 a 10
    param_grid={"model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    cv=3  # Número de divisiones para la validación cruzada
)

# Entrenamos el modelo con GridSearchCV
mod.fit(X, y)

# Guardamos los resultados de la búsqueda en un DataFrame (tabla)
df = pd.DataFrame(mod.cv_results_)
# Guardamos la tabla en un archivo CSV para revisarla después
df.to_csv('cv_results.csv', index=False)

# Mostramos ambos gráficos juntos
plt.tight_layout()  # Ajusta el espacio entre los gráficos para que no se solapen
plt.show()  # Muestra los gráficos en una ventana
