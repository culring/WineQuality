import heapq


class ClassifierSet:
    def __init__(self):
        self._model = []

    def add(self, model):
        self._model.append(model)

    def predict(self, X):
        y = []
        for sample in X:
            votes = [model.predict([sample]) for model in self._model]
            y.append(sum(votes) / len(votes))

        return y

    def get_worst_samples(self, X, y, n):
        predictions = self.predict(X)
        errors = []
        for index, (e_correct, e_predicted) in enumerate(zip(y, predictions)):
            errors.append([abs(e_correct - e_predicted), index])
        heapq.heapify(errors)

        # get n samples with the highest errors
        X_worst = [X[index] for _, index in errors[-n:]]
        y_worst = [y[index] for _, index in errors[-n:]]

        return X_worst, y_worst
