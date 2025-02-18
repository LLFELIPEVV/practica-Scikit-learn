# 📂 HERRAMIENTAS PARA MANEJAR DATOS
import numpy as np  # Calculadora avanzada para operaciones matemáticas con datos
# Organizador de datos tipo Excel (tablas y análisis básico)
import pandas as pd

# 🎨 HERRAMIENTAS PARA VISUALIZACIÓN
import matplotlib.pyplot as plt  # Pincel digital para crear gráficos básicos
import seaborn as sns  # Pincel profesional para gráficos atractivos y detallados

# 🔧 HERRAMIENTAS DE MACHINE LEARNING
# Ajustadores de escala
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, OneHotEncoder
# Clasificador tipo "vecinos cercanos"
from sklearn.neighbors import KNeighborsClassifier
# Cadena de procesos automáticos (escalado + modelo)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 🎨 Configuramos el estilo visual de todos los gráficos
# Fondo claro y colores modernos
sns.set_theme(style="whitegrid", palette="viridis")

# 🔄 MEJORAMOS LOS DATOS PARA QUE EL MODELO ENTIENDA MEJOR:
# Escalar los datos ayuda a que el modelo funcione mejor y más rápido
# (Como poner todas las medidas en la misma escala)

# 📂 Cargamos nuestros datos desde un archivo CSV
df = pd.read_csv("drawndata1.csv")

# 👀 Vistazo rápido a las primeras 3 filas de los datos
print("Primeras 3 filas de nuestros datos:")
df.head(3)
print("\nInformación técnica:")
df.info()  # Resumen de columnas y tipos de datos

# 🎯 Separamos los datos en dos partes:
X = df[['x', 'y']].values  # Coordenadas X e Y (datos para predecir)
y = df['z'] == "a"  # Etiquetas: Verdadero/Falso según la columna 'z'

# 📊 Gráfico 1: Datos originales
plt.figure(figsize=(10, 6))  # Tamaño del gráfico
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={True: "#440154", False: "#fde725"},
                edgecolor="black", alpha=0.8, s=80)
plt.title(
    "Distribución Original de los Datos\n(Colores muestran diferentes grupos)", pad=20)
plt.xlabel("Coordenada X", labelpad=10)
plt.ylabel("Coordenada Y", labelpad=10)
plt.legend(title="Grupo A", loc="upper right")
plt.show()

# 🔧 Ajustamos la escala de los datos
escalador = StandardScaler()  # Creamos un ajustador de escalas
X_new = escalador.fit_transform(X)  # Transformamos los datos

# 📊 Gráfico 2: Datos escalados
plt.figure(figsize=(10, 6))
sc = sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1], hue=y,
                     palette={True: "#21918c", False: "#fde725"},
                     edgecolor="black", alpha=0.8, s=80)
plt.title("Datos después del Escalado\n(Misma información en nueva escala)", pad=20)
plt.xlabel("Coordenada X (Escalada)", labelpad=10)
plt.ylabel("Coordenada Y (Escalada)", labelpad=10)
plt.legend(title="Grupo A", loc="upper right")
# Líneas de referencia para el punto (0,0) después del escalado
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.show()

# 🧪 Ejemplo adicional: Cómo funciona el escalado
# Generamos 1000 números aleatorios combinando dos distribuciones
datos_aleatorios = np.random.exponential(
    10, 1000) + np.random.normal(0, 1, 1000)

# 🔄 Aplicamos escalado a nuestros datos aleatorios
datos_escalados = (datos_aleatorios - np.mean(datos_aleatorios)
                   ) / np.std(datos_aleatorios)

# 📊 Gráfico 3: Histograma mejorado
plt.figure(figsize=(10, 6))
sns.histplot(datos_escalados, bins=30, kde=True, color="#440154",
             edgecolor="white", linewidth=1.2)
plt.title(
    "Distribución de Datos Escalados\n(Línea curva muestra la tendencia)", pad=20)
plt.xlabel("Valores Estandarizados (media = 0, desviación = 1)", labelpad=10)
plt.ylabel("Cantidad de Datos", labelpad=10)
plt.grid(axis='y', alpha=0.3)
plt.show()

# 🔄 TRANSFORMACIÓN DE DATOS CON MÉTODO CUANTILES
# Transformación que ordena y divide los datos en grupos iguales para mejor distribución
# Método especial para manejar valores extremos y distribuciones irregulares

# 📏 APLICAMOS LA TRANSFORMACIÓN CON 100 NIVELES DE PRECISIÓN
# Usamos 100 divisiones para mayor precisión (como reglas de medición más finas)
X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)

# 📊 CREAMOS UN GRÁFICO PROFESIONAL PARA ENTENDER LOS RESULTADOS
# Tamaño ideal para gráficos claros (10 pulgadas de ancho x 6 de alto)
plt.figure(figsize=(10, 6))

# 🎨 DISEÑAMOS EL GRÁFICO DE PUNTOS MEJORADO
scatter = sns.scatterplot(
    x=X_new[:, 0],  # Valores transformados en el eje horizontal
    y=X_new[:, 1],  # Valores transformados en el eje vertical
    hue=y,  # Colorear puntos según pertenezcan al Grupo A (Sí/No)
    # Azul para el Grupo A, Amarillo para otros
    palette={True: "#2D708E", False: "#FDE725"},
    edgecolor="black",  # Borde negro para diferenciar puntos superpuestos
    alpha=0.8,  # Transparencia para ver densidad de puntos
    s=80  # Tamaño ideal para visualización clara
)

# 🔍 PERSONALIZAMOS LA INFORMACIÓN DEL GRÁFICO
# Título descriptivo en dos líneas
plt.title(
    "Datos Reorganizados con Método Cuantiles\n(Mayor precisión en análisis)", pad=20)
# Etiqueta eje horizontal con espacio adicional
plt.xlabel("Posición X Transformada", labelpad=10)
# Etiqueta eje vertical con espacio adicional
plt.ylabel("Posición Y Transformada", labelpad=10)
plt.legend(title="¿Pertenece al Grupo A?", loc="upper right",
           labels=['No', 'Sí'])  # Cuadro explicativo de colores

# ➕ LÍNEAS GUÍA PARA MEJOR ORIENTACIÓN
# Línea horizontal central de referencia
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# Línea vertical central de referencia
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# 🟦 CUADRÍCULA ESTILIZADA PARA FACILITAR LA LECTURA
plt.grid(alpha=0.3, linestyle=':')  # Líneas punteadas suaves como guía visual

plt.show()  # Mostrar el gráfico final


# 🎯 FUNCIÓN PARA COMPARAR MÉTODOS DE ESCALADO
def plot_output(scaler):
    """
    Crea una comparación visual de 3 elementos:
    1. Datos originales
    2. Datos transformados
    3. Predicciones del modelo

    Parámetro:
    scaler: Herramienta para ajustar la escala de los datos
    """

    # 🔧 Configuramos el proceso de análisis (escalado + modelo predictivo)
    pipe = Pipeline([
        ("scale", scaler),  # Paso 1: Ajustar escala
        # Paso 2: Modelo de predicción
        ("model", KNeighborsClassifier(n_neighbors=20, weights='distance'))
    ])

    # 🧠 Entrenamos el modelo con nuestros datos
    pipe.fit(X, y)

    # 🔮 Generamos predicciones para todos los puntos
    # pred = pipe.predict(X)

    # 📊 Configuramos el lienzo para 3 gráficos
    plt.figure(figsize=(15, 5))
    sns.set_theme(style="whitegrid", palette="viridis")  # Estilo profesional

    # 🖼️ Gráfico 1: Datos Originales
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                    palette="deep", edgecolor="black")
    plt.title("Datos Originales\n(Estado natural de los datos)")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend(title="Categoría")

    # 🖼️ Gráfico 2: Datos Transformados
    plt.subplot(1, 3, 2)
    X_tfm = scaler.transform(X)
    sns.scatterplot(x=X_tfm[:, 0], y=X_tfm[:, 1], hue=y,
                    palette="deep", edgecolor="black")
    plt.title(f"Datos Transformados\n({scaler.__class__.__name__})")
    plt.xlabel("Característica 1 Escalada")
    plt.ylabel("Característica 2 Escalada")
    plt.legend().remove()

    # 🖼️ Gráfico 3: Superficie de Decisión
    plt.subplot(1, 3, 3)
    # Generamos 5000 puntos aleatorios para mapear predicciones
    X_new = np.concatenate([
        np.random.uniform(0, X[:, 0].max(), (5000, 1)),
        np.random.uniform(0, X[:, 1].max(), (5000, 1))
    ], axis=1)

    # Calculamos probabilidades de predicción
    y_proba = pipe.predict_proba(X_new)

    # Graficamos el mapa de predicciones con degradado de color
    sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1], hue=y_proba[:, 1],
                    palette="coolwarm", edgecolor="none", alpha=0.8)
    plt.title("Superficie de Decisión\n(Probabilidad de pertenencia a clase)")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend().remove()

    plt.tight_layout()
    plt.show()


# 🔄 PRIMER EXPERIMENTO: Escalado Estándar
print("=== Resultados con Escalado Estándar ===")
scaler_std = StandardScaler()
plot_output(scaler=scaler_std)

# 🔄 SEGUNDO EXPERIMENTO: Escalado por Cuantiles
print("\n=== Resultados con Escalado por Cuantiles ===")
scaler_qt = QuantileTransformer(n_quantiles=100)
plot_output(scaler=scaler_qt)

# 📂 CARGAMOS Y PREPARAMOS LOS DATOS
# Abrimos nuestro archivo de datos (como un libro de Excel digital)
df = pd.read_csv("drawndata2.csv")

# 🎯 SEPARAMOS LA INFORMACIÓN:
X = df[['x', 'y']].values  # Usamos las columnas x e y como coordenadas
y = df['z'] == "a"  # Creamos etiquetas: True para 'a', False para otros valores

# 🎨 CONFIGURACIÓN VISUAL
# Fondo claro y colores vibrantes
sns.set_theme(style="whitegrid", palette="bright")

# 📊 GRÁFICO 1: DATOS ORIGINALES
plt.figure(figsize=(10, 6))  # Tamaño profesional para mejor visualización
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={True: "#FF6B6B", False: "#4ECDC4"},
                edgecolor="black", s=80, alpha=0.9)
plt.title("Distribución Original de los Datos",
          pad=15, fontsize=14)  # Título claro
plt.xlabel("Coordenada X", labelpad=10)  # Etiqueta descriptiva
plt.ylabel("Coordenada Y", labelpad=10)
plt.legend(title="Pertenece al Grupo A", labels=[
           'No', 'Sí'])  # Leyenda explicativa
plt.show()

# 🔧 CONFIGURACIÓN DEL MODELO INTELIGENTE
pipe = Pipeline([
    # Creamos combinaciones de características (ej: x², y², xy)
    ("scale", PolynomialFeatures()),
    ("model", LogisticRegression())  # Modelo para encontrar patrones complejos
])

# 🎓 ENTRENAMOS EL MODELO
# El modelo aprende patrones y hace predicciones
pred = pipe.fit(X, y).predict(X)

# 📈 GRÁFICO 2: PREDICCIONES DEL MODELO
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=pred, palette={True: "#FF6B6B", False: "#4ECDC4"},
                          edgecolor="black", s=80, alpha=0.9)
plt.title("Límites de Decisión del Modelo", pad=15, fontsize=14)
plt.xlabel("Coordenada X", labelpad=10)
plt.ylabel("Coordenada Y", labelpad=10)

# 🖌️ Personalización adicional
# Título de leyenda actualizado
scatter.legend_.set_title("Predicción del Modelo")
for text in scatter.legend_.texts:  # Cambiamos los textos de la leyenda
    text.set_text("Grupo A" if text.get_text() == "True" else "Otros grupos")

plt.show()

# 🎯 EJEMPLO DE DATOS CATEGÓRICOS
# Creamos un arreglo de categorías (texto) que queremos convertir a números
arr = np.array(["low", "low", "high", "medium"]).reshape(-1,
                                                         1)  # Formato requerido: matriz de 1 columna
print("📋 Datos categóricos originales:")
print(arr)  # Mostramos cómo se ven los datos antes de la transformación

# 🔧 CONFIGURAMOS EL CODIFICADOR ONE-HOT
# OneHotEncoder convierte categorías en vectores numéricos (ideal para modelos de ML)
enc = OneHotEncoder(sparse_output=False,  # Devuelve una matriz densa (fácil de leer)
                    handle_unknown='ignore')  # Ignora categorías no vistas durante el entrenamiento

# 🛠️ TRANSFORMAMOS LOS DATOS
# Aprendemos las categorías y las convertimos
encoded_arr = enc.fit_transform(arr)
print("\n🔢 Datos codificados (One-Hot Encoding):")
print(encoded_arr)  # Mostramos el resultado de la transformación

# 💡 EXPLICACIÓN DEL RESULTADO:
# Cada categoría se convierte en un vector binario:
# - "low" → [1, 0, 0]
# - "medium" → [0, 1, 0]
# - "high" → [0, 0, 1]

# 🧪 PRUEBA CON UNA CATEGORÍA NUEVA
# Simulamos una categoría que no estaba en los datos originales
# Transformamos "zero" usando el codificador ya entrenado
new_category = enc.transform([["zero"]])
print("\n⚠️ Resultado para categoría nueva ('zero'):")
print(new_category)  # Muestra cómo se manejan categorías desconocidas

# 📝 NOTA IMPORTANTE:
# - handle_unknown='ignore' hace que categorías nuevas se codifiquen como [0, 0, 0]
# - Esto evita errores cuando el modelo encuentra datos no vistos durante el entrenamiento
