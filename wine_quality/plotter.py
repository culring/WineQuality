import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, clf):
        self._clf = clf

    # def plot_2D(self, bounds, title='2D plot'):
    #     plt.title(title)
    #     for x in bounds:
    #         plt.