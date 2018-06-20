class PartialModel:
    def __init__(self, model):
        self._model = model
        self._features = None

    def _extract_features(self, X):
        return [[row[feature] for feature in self._features] for row in X]

    def fit(self, X, y, features):
        self._features = features
        new_X = self._extract_features(X)
        self._model.fit(new_X, y)

    def predict(self, X):
        new_X = self._extract_features(X)
        return self._model.predict(new_X)
