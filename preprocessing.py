# 📦 Importamos herramientas para trabajar con datos y gráficos
import numpy as np  # Para cálculos matemáticos
import pandas as pd  # Para trabajar con tablas de datos (como Excel)
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones
import seaborn as sns  # Para hacer gráficos más bonitos y profesionales

# 🛠️ Importamos herramienta para ajustar escalas de los datos
from sklearn.preprocessing import StandardScaler

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
