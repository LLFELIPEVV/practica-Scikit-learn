# 📦 IMPORTACIONES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklego.preprocessing import ColumnSelector
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklego.datasets import load_chicken, make_simpleseries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklego.meta import Thresholder, GroupedPredictor, DecayEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer, mean_absolute_error, mean_squared_error

# 🎨 CONFIGURACIÓN DE ESTILO
sns.set_theme(style="whitegrid", palette="muted", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)

# =================================================================
# 🧪 CLASIFICACIÓN CON MÚLTIPLES MODELOS
# =================================================================

# 📊 GENERAMOS DATOS DE CLASIFICACIÓN
X, y = make_classification(
    n_samples=2000,
    n_features=2,
    n_redundant=0,
    random_state=21,
    class_sep=1.75,
    flip_y=0.1
)

# 🔍 VISUALIZAMOS LOS DATOS ORIGINALES
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="viridis", alpha=0.8)
plt.title("Distribución de Datos de Clasificación", pad=20)
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# 🛠️ FUNCIÓN PARA COMPARAR MODELOS


def make_plots():
    # Generamos superficie de decisión
    X_new = np.concatenate([
        np.random.uniform(X[:, 0].min(), X[:, 0].max(), (20000, 1)),
        np.random.uniform(X[:, 1].min(), X[:, 1].max(), (20000, 1))
    ], axis=1)

    # Configuramos el lienzo
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Gráfico 1: Datos originales
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=axs[0],
                    palette="viridis", alpha=0.6)
    axs[0].set_title("Datos Originales", pad=15)

    # Gráficos 2-4: Superficies de decisión
    titles = ["Regresión Logística", "Vecinos Cercanos", "Ensemble"]
    models = [clf1, clf2, clf3]

    for i, (title, model) in enumerate(zip(titles, models), 1):
        probas = model.predict_proba(X_new)[:, 1]
        sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1], hue=probas,
                        ax=axs[i], palette="viridis", alpha=0.6, legend=False)
        axs[i].set_title(title, pad=15)

    plt.tight_layout()
    plt.show()


# 🤖 ENTRENAMOS MODELOS
clf1 = LogisticRegression().fit(X, y)  # Modelo lineal
clf2 = KNeighborsClassifier(n_neighbors=10).fit(
    X, y)  # Modelo basado en distancia
clf3 = VotingClassifier(  # Ensamble de modelos
    estimators=[('lr', clf1), ('knn', clf2)],
    voting='soft',
    weights=[10.5, 2.5]
).fit(X, y)

make_plots()

# =================================================================
# 🔧 AJUSTE DE UMBRALES DE CLASIFICACIÓN
# =================================================================

# 📊 GENERAMOS DATOS CON CLÚSTERES
X, y = make_blobs(
    n_samples=1000,
    centers=[(0, 0), (1.5, 1.5)],
    cluster_std=[1, 0.5]
)

# 🎨 VISUALIZACIÓN DE UMBRALES
plt.figure(figsize=(15, 5))
models = [
    Thresholder(LogisticRegression(), threshold=0.1),
    Thresholder(LogisticRegression(), threshold=0.9)
]

for i, model in enumerate([None] + models):
    plt.subplot(1, 3, i+1)
    title = "Original" if i == 0 else f"Umbral: {model.threshold}"
    if i == 0:
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="viridis")
    else:
        preds = model.fit(X, y).predict(X)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=preds, palette="viridis")
    plt.title(title, pad=15)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2" if i == 0 else "")

plt.tight_layout()
plt.show()

# =================================================================
# 📈 MODELADO CON DATOS DE POLLOS
# =================================================================

# 🐔 CARGA Y MODELADO DE DATOS
df = load_chicken(as_frame=True)


def plot_model(model):
    model.fit(df[['diet', 'time']], df['weight'])
    metric_df = df.assign(pred=model.predict(df[['diet', 'time']]))
    mae = mean_absolute_error(metric_df['weight'], metric_df['pred'])

    plt.figure(figsize=(12, 6))
    for diet in [1, 2, 3, 4]:
        subset = metric_df[metric_df['diet'] == diet]
        sns.lineplot(x='time', y='pred', data=subset,
                     label=f'Dieta {diet}', linewidth=2.5)
    plt.title(f"Predicciones por Grupo\nMAE: {mae:.2f}", pad=15)
    plt.xlabel("Tiempo (semanas)")
    plt.ylabel("Peso (kg)")
    plt.legend()
    plt.show()


# 🔧 PIPELINE DE PROCESAMIENTO
feature_pipeline = FeatureUnion([
    ("dietas", Pipeline([
        ("selector", ColumnSelector("diet")),
        ("codificador", OneHotEncoder())
    ])),
    ("tiempo", Pipeline([
        ("selector", ColumnSelector("time")),
        ("escalador", StandardScaler())
    ]))
])

pipe = Pipeline([
    ("procesamiento", feature_pipeline),
    ("modelo", LinearRegression())
])

plot_model(pipe)

# =================================================================
# ⏳ SERIES TEMPORALES CON DECAIMIENTO
# =================================================================

# 📈 GENERAMOS Y VISUALIZAMOS SERIE TEMPORAL
yt = make_simpleseries(seed=1)
dates = pd.date_range("2000-01-01", periods=len(yt))
df = pd.DataFrame({"valor": yt, "fecha": dates}).assign(
    mes=lambda d: d.fecha.dt.month,
    indice=lambda d: d.index
)

plt.figure(figsize=(12, 4))
sns.lineplot(x="fecha", y="valor", data=df, linewidth=2)
plt.title("Serie Temporal Simulada", pad=15)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.show()

# 🤖 MODELOS DE PREDICCIÓN
mod1 = GroupedPredictor(DummyRegressor(), groups=["mes"])
mod2 = GroupedPredictor(
    DecayEstimator(
        DummyRegressor(),
        decay_func=lambda X, y: 0.9**X['indice']
    ),
    groups=["mes"]
)

mod1.fit(df[['mes']], df['valor'])
mod2.fit(df[['indice', 'mes']], df['valor'])

# 🎨 COMPARACIÓN DE PREDICCIONES
plt.figure(figsize=(12, 4))
sns.lineplot(x="fecha", y="valor", data=df, label="Real", alpha=0.6)
sns.lineplot(x=df['fecha'], y=mod1.predict(
    df[['mes']]), label="Grupos", linewidth=2)
sns.lineplot(x=df['fecha'], y=mod2.predict(df[['indice', 'mes']]),
             label="Decaimiento", linewidth=2)
plt.title("Comparación de Estrategias de Modelado", pad=15)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend()
plt.show()
