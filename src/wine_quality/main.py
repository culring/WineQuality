from sklearn import tree
from csv_data_parser import CsvDataParser
from random import sample
import matplotlib.pyplot as plt
import numpy

RED_PATH = r'..\data\winequality-red.csv'
WHITE_PATH = r'..\data\winequality-white.csv'


def get_red_white_data():
    features, X, y = CsvDataParser.get_data(RED_PATH)
    features2, X2, y2 = CsvDataParser.get_data(WHITE_PATH)
    X.extend(X2)
    y.extend(y2)

    return X, y


def split_data(X, y, training_factor=0.8):
    data = list(zip(X, y))
    shuffled_data = sample(data, len(data))
    training_data_size = int(training_factor * len(data))

    training_X = [row[0] for row in shuffled_data[:training_data_size]]
    training_y = [row[1] for row in shuffled_data[:training_data_size]]
    test_X = [row[0] for row in shuffled_data[training_data_size:]]
    test_y = [row[1] for row in shuffled_data[training_data_size:]]

    return training_X, training_y, test_X, test_y


def extract_features(X, features):
    return [[row[index] for index in features] for row in X]


def create_figure(x_label, y_label, title):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    fig.suptitle(title)


def plot_2D(clf, bounds):
    plt.scatter(bounds, [clf.predict(x) for x in bounds])
    plt.show()


def alcohol_plot(clf):
    # plot the 2D chart
    create_figure('alcohol', 'quality', 'Regression Tree model based on alcohol feature')
    plot_2D(clf, numpy.arange(10, 30, 0.001))


def measure_accuracy(clf):
    pass


def main():
    # prepare the data
    X, y = get_red_white_data()
    # X = extract_features(X, [10])
    training_X, training_y, test_X, test_y = split_data(X, y, 0.8)

    # create the classifier
    clf = tree.DecisionTreeRegressor(presort=True)
    clf.fit(training_X, training_y)

    # make the predictions
    predictions = clf.predict(test_X)
    print(all([int(x) == x for x in predictions]))

    # test accuracy
    print(clf.score(test_X, test_y))

    # check accuracy
    diff_array = [0 if pred_output != output else 1 for pred_output, output in zip(predictions, test_y)]
    n_correct = sum(diff_array)
    n_incorrect = len(diff_array) - n_correct

    print(n_correct/(n_correct + n_incorrect) * 100)
    # print(clf.feature_importances_)

    plt.contourf([1, 2], [1, 2], [[1, 1], [2, 2]])


if __name__ == '__main__':
    main()
