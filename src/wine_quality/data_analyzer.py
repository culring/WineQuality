from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

from data_modifier import DataModifier
from csv_data_parser import CsvDataParser


class DataAnalyzer:
    @staticmethod
    def extract_class(X, y, cat=0):
        new_X, new_y = [], []
        for x_val, y_val in zip(X, y):
            if y_val != cat:
                continue
            new_X.append(x_val)
            new_y.append(y_val)
        return new_X, new_y

    @staticmethod
    def get_feature_importances(X, y):
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        return clf.feature_importances_

    @staticmethod
    def pretty_print(to_print, prefix_line='', suffix_line=''):
        if prefix_line:
            print('--------------------', prefix_line, '--------------------', sep='')
        print(to_print)
        if suffix_line:
            print('--------------------', suffix_line, '--------------------', sep='')

    @staticmethod
    def wines_per_quality(y):
        qualities = 11 * [0]
        for sample in y:
            qualities[sample] += 1
        return qualities

    @classmethod
    def print_wines_per_quality(cls, y):
        wines_per_quality_list = cls.wines_per_quality(y)
        for quality, num in enumerate(wines_per_quality_list):
            cls.pretty_print(num, prefix_line=f'quality {quality}')


if __name__ == '__main__':
    features, X, y = CsvDataParser.get_data(r'E:\studia\sem6\UM\projekt\data\winequality-red.csv')
    features2, X2, y2 = CsvDataParser.get_data(r'E:\studia\sem6\UM\projekt\data\winequality-white.csv')
    X.extend(X2)
    y.extend(y2)
    DataModifier.normalize_data(X)

    # for i in range(10):
    #     pretty_print([round(val, 3) for val in X[i]], prefix_line=f'Sample {i}')

    # print('Anomaly detection values:')
    # print('1079: ', X[1079])
    # print('1081: ', X[1081])

    print(list(DataAnalyzer.get_feature_importances(X, y)))

    pca = PCA(n_components=10)
    pca.fit(X)
    # print([round(x, 4) for x in pca.explained_variance_ratio_])
    X = pca.transform(X)

    print(list(DataAnalyzer.get_feature_importances(X, y)))
    print(sum(list(DataAnalyzer.get_feature_importances(X, y))))

    # print_wines_per_quality(y)
    # plt.hist(y, color='r', bins=range(11))
    # plt.xticks(range(11))
    # plt.title('Red wines - quality histogram')
    # plt.show()

    # colors for drawing charts
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # draw one chart per class
    # min_x = min([sample[0] for sample in X])-1
    # min_y = min([sample[1] for sample in X])-1
    # max_x = max([sample[0] for sample in X])+1
    # max_y = max([sample[1] for sample in X])+1
    # for i in range(2, 10):
    #     new_X, new_y = extract_class(X, y, cat=i)
    #     plt.title(f'PCA features (category {i})')
    #     plt.scatter(x=[x[0] for x in new_X], y=[x[1] for x in new_X], c=colors[i-3])
    #     axes = plt.gca()
    #     axes.set_xlim([min_x, max_x])
    #     axes.set_ylim([min_y, max_y])
    #     plt.show()
    #
    # # # idxs = []
    # # # for i, x_val in enumerate(X):
    # # #     if x_val[0] > 3.5 and y[i] == 7:
    # # #         idxs.append(i)
    # # # print('Anomaly detection: ', idxs)
    # #
    # draw one chart
    # n_col = 0
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # cat_set = [6, 7, 8]
    # for i in cat_set:
    #     new_X, new_y = extract_class(X, y, cat=i)
    #     if not len(new_X):
    #         continue
    #     axis = 3*[0]
    #     for j in range(3):
    #         axis[j] = [x[j] for x in new_X]
    #     ax.scatter(axis[0], axis[1], zs=axis[2], c=colors[i-3], label=f'category {i}')
    #     # plt.scatter(x=[x[0] for x in new_X], y=[x[1] for x in new_X], c=colors[n_col], label=f'category {i}')
    #     n_col += 1
    #
    # # plt.title(f'PCA features (red and white wines)')
    # plt.title(f'PCA features (red and white wines)')
    # plt.legend()
    # # axes = plt.gca()
    # # axes.set_xlim([min_x, max_x])
    # # axes.set_ylim([min_y, max_y])
    # plt.show()

    # clf = ExtraTreesClassifier()
    # clf = clf.fit(X, y)
    # nlargest = heapq.nlargest(5, clf.feature_importances_.tolist())
    # for i, importance in enumerate(clf.feature_importances_):
    #     print(f'feature {i}:\t{round(importance, 3)}' + ('\t<---' if importance in nlargest else ''))
