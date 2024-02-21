from time import time

import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from perceptron import Perceptron
from adaline import AdalineGD
from from_books.perceptron import NumpyPerceptron
from from_books.adaline import NumpyAdalineGD

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

# Fragment from book to get data in numpy format
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None,
                 encoding='utf-8')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

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
model = perceptron.fit(X, y)
t2 = time()
print('Weights:', model.w_)
print(t2 - t1)

"""It is 0.0572s for my perceptron and 0.0640s for perceptron from book
My perceptron is slightly faster. Weights are the same for at least 5 significant digits
some approximation error during operations have occurred
UPDATE: I make a mistake, by not using numpy arrays as a input  for NumpyPerceptron
So a time converting Polars data frame to a numpy also was measured.
After repair this problem I see a execution time for NumpyPerceptron is 0.0387, what is better result, than polars"""


print('_______________________ Polars adaline _______________________________')
t1 = time()
adaline = AdalineGD(n_iterations=50, eta=0.01)
model = adaline.fit(features, target)
t2 = time()
print('Weights:', list(model.weights))
print(t2 - t1)

print('_______________________ Numpy adaline _______________________________')
t1 = time()
adaline = NumpyAdalineGD(n_iter=50, eta=0.01)
model = adaline.fit(X, y)
t2 = time()
print('Weights:', model.w_)
print(t2 - t1)

"""Some conclusions:
Execution time for implementation in numpy is smaller, than polars implementation.
It is especially visible, for second example, where execution time is order of magnitude smaller
It is necessary to check this on larger data sets, perhaps converting polars to numpy in adaline
have big overhead for relatively small data sets. Nevertheless Polar lack some numpy functionality
and I often have to use numpy functions, so it can't work faster than numpy."""

