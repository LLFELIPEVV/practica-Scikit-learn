import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn. model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, make_scorer

df = pd.read_csv('creditcard.csv')[:80_000]
print(df.head(3))

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values
print(f"Shapes of X={X.shape} y={y.shape}, #Fraud Cases {y.sum()}")

# EL class_weight es un diccionario que permite establecer el peso que tendra cada clase del DataFrame, en este caso la clase 0 es sin fraude y la clase 1 es fraude.
mod = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)
print(mod.fit(X, y).predict(X).sum())

lr = LogisticRegression()
# help(lr.score)

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},
    cv=4,
    n_jobs=-1
)

print(grid.fit(X, y))

# Para decidir entre los dos es bueno pensar si nos preocupa mas los falsos positivos o los falsos negativos
# Precision dice dado que predigo el fraude que tan preciso soy.
presicion = precision_score(y, grid.predict(X))
# Recall dice si obtuve todos los casos de fraude
recall = recall_score(y, grid.predict(X))

print(f"Precision: {presicion}, Recall: {recall}")

new_df = pd.DataFrame(grid.cv_results_)

print(new_df)

mod = IsolationForest().fit(X)
pre = np.where(mod.predict(X) == -1, 1, 0)
print(pre)


def outlier_precision(mod, X, y):
    preds = mod.predict(X)
    return precision_score(y, np.where(preds == -1, 1, 0))


def outlier_recall(mod, X, y):
    preds = mod.predict(X)
    return recall_score(y, np.where(preds == -1, 1, 0))


def min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)


grid = GridSearchCV(
    estimator=IsolationForest(),
    param_grid={'contamination': np.linspace(0.001, 0.02, 10)},
    scoring={'precision': outlier_precision,
             'recall': outlier_recall},
    refit='precision',
    cv=5,
    n_jobs=-1
)

print(grid.fit(X, y))

# s = make_scorer(min_recall_precision)
# help(s)

new_df = pd.DataFrame(grid.cv_results_)

print(new_df)

plt.figure(figsize=(12, 4))
for score in ['mean_test_recall', 'mean_test_precision']:
    plt.plot(new_df['param_contamination'],
             new_df[score], label=score)
    plt.legend()
    plt.show()

plt.figure(figsize=(12, 4))
for score in ['mean_test_recall', 'mean_test_precision']:
    plt.scatter(new_df['param_contamination'],
                new_df[score], label=score)
plt.legend()
plt.show()
