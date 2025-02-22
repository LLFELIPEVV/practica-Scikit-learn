# 📦 IMPORTACIONES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Para gráficos más profesionales

from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, make_scorer

# 🎨 CONFIGURACIÓN DE ESTILO PARA GRÁFICOS
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.figure(figsize=(10, 6))  # Tamaño por defecto para gráficos

# 📂 CARGA Y EXPLORACIÓN DE DATOS
# Cargamos solo las primeras 80,000 transacciones
df = pd.read_csv('creditcard.csv')[:80_000]
print("🔍 Primeras 3 transacciones:")
print(df.head(3))  # Mostramos muestra inicial de datos

# 🎯 PREPARACIÓN DE DATOS PARA MODELADO
# Eliminamos columnas no relevantes
X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values  # Variable objetivo: 1 = fraude, 0 = transacción normal
print(f"\n📐 Dimensiones: X={X.shape} y={y.shape}")
print(f"⚠️ Casos de fraude detectados: {y.sum()}")

# 🔄 MODELO DE REGRESIÓN LOGÍSTICA CON PESOS BALANCEADOS
print("\n🔧 Entrenando modelo inicial...")
# Mayor peso a casos de fraude
mod = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)
predicciones = mod.fit(X, y).predict(X)
print(f"🔮 Predicciones de fraude iniciales: {predicciones.sum()}")

# 🧪 OPTIMIZACIÓN DE HIPERPARÁMETROS CON BUSQUEDA EN REJILLA
print("\n⚙️ Optimizando pesos de clases...")
grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    # Probamos 3 pesos diferentes
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},
    cv=4,  # Validación cruzada con 4 divisiones
    n_jobs=-1  # Usar todos los núcleos del procesador
)
grid.fit(X, y)
print("✅ Optimización completada!")

# 📈 MÉTRICAS DE RENDIMIENTO
# Exactitud en detectar fraudes reales
presicion = precision_score(y, grid.predict(X))
# Capacidad de encontrar todos los fraudes
recall = recall_score(y, grid.predict(X))
print(f"\n📊 Resultados:")
print(
    f"• Precisión: {presicion:.2%} (Transacciones marcadas como fraude que realmente lo son)")
print(f"• Recall: {recall:.2%} (Fraudes reales detectados)")

# 🕵️ MODELO DE DETECCIÓN DE ANOMALÍAS
print("\n🔍 Entrenando modelo de detección de anomalías...")
mod = IsolationForest().fit(X)  # Modelo no supervisado
pre = np.where(mod.predict(X) == -1, 1, 0)  # Convertir predicciones a 0/1
print(f"🔮 Posibles anomalías detectadas: {pre.sum()}")

# 🎯 FUNCIONES PERSONALIZADAS PARA EVALUACIÓN


def outlier_precision(mod, X, y):
    """Calcula precisión para detección de anomalías"""
    preds = mod.predict(X)
    return precision_score(y, np.where(preds == -1, 1, 0))


def outlier_recall(mod, X, y):
    """Calcula recall para detección de anomalías"""
    preds = mod.predict(X)
    return recall_score(y, np.where(preds == -1, 1, 0))


# � OPTIMIZACIÓN DE MODELO DE ANOMALÍAS
print("\n⚙️ Ajustando sensibilidad del detector...")
grid = GridSearchCV(
    estimator=IsolationForest(),
    param_grid={'contamination': np.linspace(
        0.001, 0.02, 10)},  # Rangos de contaminación
    scoring={'precision': outlier_precision, 'recall': outlier_recall},
    refit='precision',  # Priorizar precisión al seleccionar mejor modelo
    cv=5,  # Validación cruzada más estricta
    n_jobs=-1
)
grid.fit(X, y)
print("✅ Optimización completada!")

# 📊 VISUALIZACIÓN DE RESULTADOS (MEJORADA CON SEABORN)
new_df = pd.DataFrame(grid.cv_results_)

# Gráfico de líneas para tendencias
plt.figure(figsize=(12, 6))
sns.lineplot(data=new_df, x='param_contamination', y='mean_test_recall',
             label='Recall (Detección de fraudes)', linewidth=2.5, color='#2ecc71')
sns.lineplot(data=new_df, x='param_contamination', y='mean_test_precision',
             label='Precisión (Exactitud)', linewidth=2.5, color='#e74c3c')
plt.title(
    "Relación entre Precisión y Recall\n al variar la sensibilidad del detector", pad=20)
plt.xlabel("Nivel de Contaminación Ajustado", labelpad=12)
plt.ylabel("Puntuación", labelpad=12)
plt.legend()
plt.show()

# Gráfico de dispersión para puntos individuales
plt.figure(figsize=(12, 6))
sns.scatterplot(data=new_df, x='param_contamination', y='mean_test_recall',
                label='Recall', s=100, color='#2ecc71', edgecolor='black')
sns.scatterplot(data=new_df, x='param_contamination', y='mean_test_precision',
                label='Precisión', s=100, color='#e74c3c', edgecolor='black')
plt.title("Comparación Directa de Métricas por Configuración", pad=20)
plt.xlabel("Nivel de Contaminación Ajustado", labelpad=12)
plt.ylabel("Puntuación", labelpad=12)
plt.legend()
plt.show()
