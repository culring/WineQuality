from random import sample

import numpy as np


class DataModifier:
    @staticmethod
    def normalize_data(X, method='std_dev'):
        n_features = len(X[0])
        n_samples = len(X)
        for i in range(n_features):
            samples = [sample[i] for sample in X]
            if method == 'min_max':
                max_val, min_val = max(samples), min(samples)
                range_factor = max_val - min_val
            elif method == 'std_dev':
                range_factor = np.std(samples)
            else:
                raise ValueError('Wrong method parameter passed.')
            mean_val = sum(samples) / n_samples
            samples = [(sample - mean_val) / range_factor for sample in samples]
            for j, sample in enumerate(X):
                sample[i] = samples[j]

    @staticmethod
    def split_data(X, y, training_factor=0.8):
        data = list(zip(X, y))
        shuffled_data = sample(data, len(data))
        training_data_size = int(training_factor * len(data))

        training_X = [row[0] for row in shuffled_data[:training_data_size]]
        training_y = [row[1] for row in shuffled_data[:training_data_size]]
        test_X = [row[0] for row in shuffled_data[training_data_size:]]
        test_y = [row[1] for row in shuffled_data[training_data_size:]]

        return training_X, training_y, test_X, test_y
