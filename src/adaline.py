import polars as pl
import numpy as np


# Binding numpy dot function to polars
def dot(left_df: pl.DataFrame, right_df: pl.DataFrame):
    return pl.from_numpy(np.dot(left_df, right_df))
pl.DataFrame.dot = dot


class AdalineGD:
    def __init__(self, eta=0.01, n_iterations=50, random_state=42):
        self.eta = eta
        self.n_iterations = n_iterations
        self.__random_generator = np.random.RandomState(random_state)
        self.weights = None

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
        activation_force = features.dot(self.weights) + self.bias
        return activation_force.select(pl.when(pl.col('column_0') >= 0.5).then(1).otherwise(0).alias('prediction'))

    def score(self, features, target):
        if self.weights is None:
            raise Exception('Fit method haven\'t been used!')
        results = self.predict(features)
        correct_predictions = 0
        for prediction, true_result in zip(results.iter_rows(), target.iter_rows()):
            if prediction == true_result:
                correct_predictions += 1
        score = correct_predictions / len(target)
        return score


