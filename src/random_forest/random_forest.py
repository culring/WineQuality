import random

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from .partial_model import PartialModel
from .classifier_set import ClassifierSet

from sklearn.metrics import mean_squared_error


class RandomForest(BaseEstimator):
    def __init__(self, n_trees: int = 1, samples='all', n_features='all'):
        if type(samples) == float and (samples <= 0.0 or samples > 1.0):
            raise ValueError('Parameter \'n_samples\' has to be either set to \'all\' or be a positive integer value.')
        elif type(samples) != float and samples != 'all':
            raise ValueError('Parameter \'n_samples\' has to be either set to \'all\' or be a positive integer value.')
        if n_features != 'all' and type(n_features) != int:
            raise ValueError('Parameter \'n_features\' has to be either set to \'all\' or be a positive integer value.')

        self._M = ClassifierSet()
        self.n_trees = n_trees
        self.samples = samples
        self.n_features = n_features

    def _generate_random_samples(self, X, y, n_samples):
        indices = random.sample(range(len(X)), n_samples)
        return [X[i] for i in indices], [y[i] for i in indices]

    def _generate_random_sequence(self, n, end, begin=0):
        return random.sample(range(begin, end), n)

    def fit(self, X, y):
        if self.samples == 'all':
            n_samples = len(X)
        else:
            n_samples = int(self.samples*len(X))

        if self.n_features == 'all':
            self.n_features = len(X[0])

        SX, Sy = self._generate_random_samples(X, y, n_samples)
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor()
            partial_model = PartialModel(tree)
            features = self._generate_random_sequence(self.n_features, len(SX[0]))
            partial_model.fit(SX, Sy, features)
            self._M.add(partial_model)
            SX, Sy = self._M.get_worst_samples(X, y, n_samples)

    def predict(self, X):
        return self._M.predict(X)
