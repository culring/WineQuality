import csv


class CsvDataParser:
    @staticmethod
    def get_data(path):
        with open(path) as file:
            reader = csv.reader(file, delimiter=';')
            data = list(reader)
            for i, line in enumerate(data[1:], 1):
                data[i] = [float(x) if i != len(line) - 1 else int(x) for i, x in enumerate(line)]
            features = data[0]
            X = [line[:-1] for line in data[1:]]
            y = [line[-1] for line in data[1:]]
            return features, X, y
