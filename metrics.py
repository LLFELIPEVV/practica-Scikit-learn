import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('creditcard.csv')[:80_000]
print(df.head(3))

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values
print(f"Shapes of X={X.shape} y={y.shape}, #Fraud Cases {y.sum()}")

# EL class_weight es un diccionario que permite establecer el peso que tendra cada clase del DataFrame, en este caso la clase 0 es sin fraude y la clase 1 es fraude.
mod = LogisticRegression(class_weight={0:1, 1:2}, max_iter=1000)
print(mod.fit(X, y).predict(X).sum())
