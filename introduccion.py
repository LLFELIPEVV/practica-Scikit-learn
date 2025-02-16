# 📦 Importamos herramientas para trabajar con datos y gráficos
import matplotlib.pyplot as plt  # Para dibujar gráficos
import seaborn as sns  # Para hacer gráficos más claros y profesionales
import pandas as pd  # Para organizar datos en tablas como Excel

# 🛠️ HERRAMIENTAS DE APRENDIZAJE AUTOMÁTICO:
from sklearn.datasets import load_diabetes  # Datos de pacientes con diabetes
# Modelo de predicción tipo "vecinos cercanos"
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler  # Ajustador de escalas numéricas
from sklearn.pipeline import Pipeline  # Cadena de procesos automáticos
from sklearn.model_selection import GridSearchCV  # Buscador de mejores ajustes

# 📝 PASOS BÁSICOS DEL APRENDIZAJE AUTOMÁTICO:
# 1. Obtener datos (información de entrada).
# 2. Separar los datos en características (X) y lo que queremos predecir (y).
# 3. Crear un modelo que pueda aprender de los datos.
# 4. Entrenar el modelo con los datos.
# 5. Usar el modelo para hacer predicciones.

# 📂 CARGAMOS LOS DATOS DE DIABETES
# X = Características de los pacientes (edad, peso, análisis de sangre, etc.)
# y = Progresión de la diabetes (lo que queremos predecir)
X, y = load_diabetes(return_X_y=True)

# 🖨️ MOSTRAMOS LOS DATOS CARGADOS
print("📊 Datos de entrada (características):")
print(X)  # Tabla con números que representan las características
print("\n🎯 Lo que queremos predecir:")
print(y)  # Lista de valores que representan la progresión de la diabetes

# 🎨 CONFIGURAMOS EL ESTILO DE LOS GRÁFICOS
# Fondo con cuadrícula para mejor visualización
sns.set_theme(style="whitegrid")

# --- 🔍 PRIMER EXPERIMENTO: Sin normalizar los datos ---
# Creamos un modelo básico de predicción (KNN: Vecinos más cercanos)
modelo_basico = KNeighborsRegressor()

# 🎓 ENTRENAMOS EL MODELO CON LOS DATOS
modelo_basico.fit(X, y)

# 🔮 HACEMOS PREDICCIONES CON EL MODELO BÁSICO
predicciones_sin_ajuste = modelo_basico.predict(X)

# 📊 CREAMOS GRÁFICOS PARA COMPARAR PREDICCIONES Y VALORES REALES
plt.figure(figsize=(15, 5))  # Tamaño del lienzo para los gráficos

# 📈 GRÁFICO 1: Predicciones sin normalizar los datos
plt.subplot(1, 3, 1)  # Posición 1 en una fila de 3 gráficos
sns.scatterplot(x=predicciones_sin_ajuste, y=y,
                alpha=0.6, color='blue')  # Puntos azules
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia ideal
plt.xlabel("Predicciones (sin normalizar)")  # Etiqueta del eje X
plt.ylabel("Valores reales")  # Etiqueta del eje Y
plt.title("Predicciones sin normalización")  # Título del gráfico

# --- 🧪 SEGUNDO EXPERIMENTO: Normalizando los datos ---
# Mejoramos el proceso: primero normalizamos los datos, luego hacemos predicciones
tuberia_mejorada = Pipeline([
    ("escalador", StandardScaler()),  # Paso 1: Normalizar los datos
    ("modelo", KNeighborsRegressor(n_neighbors=1))  # Paso 2: Predecir con KNN
])

# 🎓 ENTRENAMOS EL MODELO MEJORADO
tuberia_mejorada.fit(X, y)

# 🔮 HACEMOS PREDICCIONES CON EL MODELO MEJORADO
predicciones_con_ajuste = tuberia_mejorada.predict(X)

# 📈 GRÁFICO 2: Predicciones con datos normalizados
plt.subplot(1, 3, 2)  # Posición 2 en una fila de 3 gráficos
sns.scatterplot(x=predicciones_con_ajuste, y=y,
                alpha=0.6, color='green')  # Puntos verdes
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia ideal
plt.xlabel("Predicciones (con normalización)")
plt.ylabel("Valores reales")
plt.title("Predicciones con normalización")

# --- 🚀 TERCER EXPERIMENTO: Optimizando el modelo ---
# Buscamos automáticamente la mejor configuración para el modelo
print("\n⚙️ Parámetros que podemos ajustar:", tuberia_mejorada.get_params())

# Configuramos el buscador de mejores ajustes (GridSearchCV)
buscador_optimizador = GridSearchCV(
    estimator=tuberia_mejorada,  # Usamos el modelo mejorado
    # Probamos diferentes valores
    param_grid={"modelo__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    cv=3  # Validación cruzada: prueba cada opción 3 veces
)

# 🎓 ENTRENAMOS EL BUSCADOR PARA ENCONTRAR LA MEJOR CONFIGURACIÓN
buscador_optimizador.fit(X, y)

# 💾 GUARDAMOS LOS RESULTADOS EN UN ARCHIVO CSV
resultados_experimentos = pd.DataFrame(buscador_optimizador.cv_results_)
resultados_experimentos.to_csv('resultados_optimizacion.csv', index=False)

# 🏆 MOSTRAMOS LOS MEJORES PARÁMETROS ENCONTRADOS
print(f"✅ Mejores parámetros: {buscador_optimizador.best_params_}")

# 🔧 ACTUALIZAMOS EL MODELO CON LOS MEJORES PARÁMETROS
tuberia_mejorada.set_params(**buscador_optimizador.best_params_)
tuberia_mejorada.fit(X, y)  # Reentrenamos el modelo con la mejor configuración

# 🔮 HACEMOS PREDICCIONES CON EL MODELO OPTIMIZADO
predicciones_optimizadas = tuberia_mejorada.predict(X)

# 📈 GRÁFICO 3: Predicciones con la mejor configuración
plt.subplot(1, 3, 3)  # Posición 3 en una fila de 3 gráficos
sns.scatterplot(x=predicciones_optimizadas, y=y, alpha=0.6,
                color='purple')  # Puntos morados
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',
         linestyle='--')  # Línea de referencia ideal
plt.xlabel("Predicciones (mejor configuración)")
plt.ylabel("Valores reales")
plt.title("Predicciones optimizadas")

# 🖼️ MOSTRAMOS TODOS LOS GRÁFICOS JUNTOS
plt.tight_layout()  # Ajustamos el espacio entre gráficos
plt.show()  # Abrimos la ventana con los gráficos

# 📚 INFORMACIÓN ADICIONAL SOBRE LOS DATOS
diabetes = load_diabetes()
print("\nDescripción técnica del conjunto de datos:")
print(diabetes.DESCR)  # Explicación médica de las variables
