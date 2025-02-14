# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt  # Para graficar los resultados
import seaborn as sns  # Para gráficos más atractivos

from sklearn.datasets import load_diabetes  # Para cargar el conjunto de datos
# Para usar el modelo de regresión KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler  # Para escalar los datos
from sklearn.pipeline import Pipeline  # Para crear una tubería de procesamiento

# En scikit-learn, el flujo de trabajo general es el siguiente:
# 1. Cargar o generar los datos.
# 2. Dividir los datos en dos partes: X (características) e y (etiquetas o valores a predecir).
# 3. Crear un modelo de aprendizaje automático.
# 4. Entrenar el modelo con los datos (X, y).
# 5. Usar el modelo entrenado para hacer predicciones.

# Cargamos el conjunto de datos de diabetes
# `X` contiene las características (datos que se usan para hacer predicciones).
# `y` contiene las etiquetas (lo que queremos predecir).
X, y = load_diabetes(return_X_y=True)

# Mostramos los datos cargados
# Muestra las características (datos de entrada)
print(f"Características (X): {X}")
# Muestra las etiquetas (lo que queremos predecir)
print(f"Etiquetas (y): {y}")

# Configuramos el estilo de los gráficos con Seaborn
sns.set_theme(style="whitegrid")  # Estilo de fondo con cuadrícula

# --- Primer caso: Sin escalado de datos ---
# Creamos el modelo de regresión K-Nearest Neighbors (KNN)
# KNN es un algoritmo que predice el valor de un punto basándose en los valores de los puntos más cercanos.
mod = KNeighborsRegressor()

# Entrenamos el modelo con los datos
# El método `fit` hace que el modelo "aprenda" de los datos (X, y).
mod.fit(X, y)

# Usamos el modelo entrenado para hacer predicciones sobre los mismos datos (X)
# `pred_sin_escalado` contendrá las predicciones que el modelo hace para cada valor de X.
pred_sin_escalado = mod.predict(X)

# Graficamos los resultados sin escalado
# Usamos un gráfico de dispersión para comparar las predicciones (`pred`) con los valores reales (`y`).
plt.figure(figsize=(10, 5))  # Tamaño del gráfico
plt.subplot(1, 2, 1)  # Primer subgráfico (1 fila, 2 columnas, posición 1)

sns.scatterplot(x=pred_sin_escalado, y=y, alpha=0.6,
                color='blue')  # Gráfico de dispersión

plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia (y = x)
plt.xlabel("Predicciones (sin escalado)")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
# Título del gráfico
plt.title("Predicciones vs Valores reales (sin escalado)")

# --- Segundo caso: Con escalado de datos ---
# Es mejor transformar los datos antes de pasarselos al modelo.
# El nuevo orden seria pasar datos --> scale --> KNN --> prediccion
# Por lo que ahora el modelo contiene tanto el scale como el KNN
# Creamos una tubería (pipeline) que primero escala los datos y luego aplica el modelo KNN
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))  # Normalmente esta en 5
])

# Entrenamos la tubería con los datos
pipe.fit(X, y)

# Hacemos predicciones sobre los mismos datos (X)
pred_con_escalado = pipe.predict(X)

# Graficamos los resultados con escalado
plt.subplot(1, 2, 2)  # Segundo subgráfico (1 fila, 2 columnas, posición 2)
sns.scatterplot(x=pred_con_escalado, y=y, alpha=0.6,
                color='green')  # Gráfico de dispersión
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia (y = x)
plt.xlabel("Predicciones (con escalado)")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
# Título del gráfico
plt.title("Predicciones vs Valores reales (con escalado)")

# --- Tercer caso: 
# Los datos se dividen en tres partes dos de entrenamiento y uno de prediccion, pero en cada intento esas partes intercambian sus roles para una mejor verificacion.
# Por esta razon la definicion de modelo vuelve a aumentar.
# Se necesita usar un nuevo objeto llamado GridSearchCV para buscar los mejores hiperparametros.



# Mostramos ambos gráficos juntos
plt.tight_layout()  # Ajusta el espacio entre los gráficos
plt.show()  # Muestra los gráficos
