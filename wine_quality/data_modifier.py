import numpy as np
from enum import Enum, auto


class DataModifier:
    class Method(Enum):
        STD_DEV = auto()
        MIN_MAX = auto()

    @classmethod
    def normalize_data(cls, X, method=Method.STD_DEV):
        n_features = len(X[0])
        n_samples = len(X)
        for i in range(n_features):
            samples = [sample[i] for sample in X]
            if method == cls.Method.MIN_MAX:
                max_val, min_val = max(samples), min(samples)
                range_factor = max_val - min_val
            else:
                range_factor = np.std(samples)
            mean_val = sum(samples) / n_samples
            samples = [(sample - mean_val) / range_factor for sample in samples]
            for j, sample in enumerate(X):
                sample[i] = samples[j]
