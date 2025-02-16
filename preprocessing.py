# 📦 Importamos herramientas para trabajar con datos y gráficos
import numpy as np  # Para cálculos matemáticos
import pandas as pd  # Para trabajar con tablas de datos (como Excel)
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones
import seaborn as sns  # Para hacer gráficos más bonitos y profesionales

# 🛠️ Importamos herramienta para ajustar escalas de los datos
from sklearn.preprocessing import StandardScaler, QuantileTransformer

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
