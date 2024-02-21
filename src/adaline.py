import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split


# Binding numpy dot function to polars
def dot(left_df: pl.DataFrame, right_df: pl.DataFrame):
    return pl.from_numpy(np.dot(left_df, right_df))
pl.DataFrame.dot = dot


class AdalineGD:
    def __init__(self, eta=0.01, n_iterations=50, random_state=42):
        self.eta = eta
        self.n_iterations = n_iterations
        self.__random_generator = np.random.RandomState(random_state)

    def fit(self, features: pl.DataFrame, target):
        self.weights = pl.DataFrame(self.__random_generator.normal(loc=0., scale=0.01, size=features.width))
        self.bias = 0.
        self.losses = [0] * self.n_iterations

        for i in range(self.n_iterations):
            activation_force = features.dot(self.weights) + self.bias
            errors = (target - activation_force)
            self.weights += self.eta * 2.0 * features.transpose().dot(errors) / features.shape[0]
            self.bias += self.eta * 2.0 * errors.mean().item()
            loss = errors.select(pl.col('target') ** 2).mean().item()
            self.losses[i] = loss
        return self

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        activation_force = self.weights * features + self.bias
        return activation_force.select(pl.when(pl.col('column_1') >= 0.5).then(1).otherwise(0).alias('prediction'))


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


