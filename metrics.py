import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn. model_selection import GridSearchCV

df = pd.read_csv('creditcard.csv')[:80_000]
print(df.head(3))

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values
print(f"Shapes of X={X.shape} y={y.shape}, #Fraud Cases {y.sum()}")

# EL class_weight es un diccionario que permite establecer el peso que tendra cada clase del DataFrame, en este caso la clase 0 es sin fraude y la clase 1 es fraude.
mod = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)
print(mod.fit(X, y).predict(X).sum())

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},
    cv=4,
    n_jobs=-1
)

print(grid.fit(X, y))

new_df = pd.DataFrame(grid.cv_results_)

print(new_df)
