from time import time

import polars as pl
from perceptron import Perceptron

from from_books.Perceptron import NumpyPerceptron
from sklearn.model_selection import train_test_split

"""Comparisons of my perceptron implementation, and implementation from book
using classical iris dataset """


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pl.read_csv(data_url, has_header=False)

data = data.drop_nulls()
data = data.limit(100)

target = data.select(pl.when(pl.col('column_5') == 'Iris-setosa').then(0).otherwise(1).alias('target'))
features = data.select(pl.col('column_1', 'column_3')).rename({'column_1': 'Sepal length',  'column_3': 'Petal length'})


train_features, test_features, train_target, test_target = train_test_split(features,
                                                                            target,
                                                                            stratify=target,
                                                                            random_state=42,
                                                                            train_size=0.75)

print('_______________________ Polars perceptron _______________________________')
t1 = time()
perceptron = Perceptron(n_iterations=50)
model = perceptron.fit(features, target)
t2 = time()
print('Weights:', list(model.weights))
print(t2 - t1)

print('_______________________ Numpy perceptron _______________________________')

t1 = time()
perceptron = NumpyPerceptron(n_iter=50)
model = perceptron.fit(features.to_numpy(), target.to_numpy())
t2 = time()
print('Weights:', model.w_)
print(t2 - t1)

"""It is 0.0572s for my perceptron and 0.0640s for perceptron from book
My perceptron is slightly faster. Weights are the same for at least 5 significant digits
some approximation error during operations have occurred"""
