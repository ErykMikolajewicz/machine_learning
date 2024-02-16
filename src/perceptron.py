import polars as pl
import numpy as np


class Perceptron:

    def __init__(self, eta=0.01, n_iterations=50, random_state=42):
        self.eta = eta
        self.n_iterations = n_iterations
        # lack of random generating functionality in polars is frustrating
        self.__random_generator = np.random.RandomState(random_state)

    def fit(self, features, target):
        self.weights = pl.Series(self.__random_generator.normal(loc=0., scale=0.01, size=features.width))
        self.bias = 0.
        self.errors = [0] * self.n_iterations
        target = pl.Series(target)
        transposed_features = features.transpose()

        for i in range(self.n_iterations):
            for features_row, target_value in zip(transposed_features, target):
                predicted_value = self.predict(features_row)
                predict_result = target_value - predicted_value
                update = self.eta * predict_result
                self.weights += features_row * update
                self.bias += update
                self.errors[i] += abs(predict_result)
        return self

    def predict(self, features: pl.Series) -> int:
        activation_force = self.weights.dot(features) + self.bias
        return 1 if activation_force > 0 else 0
