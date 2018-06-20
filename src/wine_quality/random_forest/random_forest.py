import random

from sklearn.tree import DecisionTreeRegressor

from .partial_model import PartialModel
from .classifier_set import ClassifierSet


class RandomForest:
    def __init__(self):
        self._M = ClassifierSet()

    def _generate_random_samples(self, X, y, n_samples):
        indices = random.sample(range(len(X)), n_samples)
        return [X[i] for i in indices], [y[i] for i in indices]

    def _generate_random_sequence(self, n, end, begin=0):
        return random.sample(range(begin, end), n)

    def fit(self, X, y, n_trees: int = 1, n_samples='all', n_features='all'):
        if n_samples == 'all':
            n_samples = len(X)
        elif type(n_samples) != int:
            raise ValueError('Parameter \'n_samples\' has to be either set to \'all\' or be a positive integer value.')

        if n_features == 'all':
            n_features = len(X[0])
        elif type(n_features) != int:
            raise ValueError('Parameter \'n_features\' has to be either set to \'all\' or be a positive integer value.')

        SX, Sy = self._generate_random_samples(X, y, n_samples)
        for _ in range(n_trees):
            tree = DecisionTreeRegressor()
            partial_model = PartialModel(tree)
            features = self._generate_random_sequence(n_features, len(SX[0]))
            partial_model.fit(SX, Sy, features)
            self._M.add(partial_model)
            SX, Sy = self._M.get_worst_samples(X, y, n_samples)

    def predict(self, X):
        return self._M.predict(X)
