class Accuracy:
    @staticmethod
    def r_squared(pred_y, y):
        mean_y = sum(y) / len(y)
        pred_deviation = sum([(pred_value - mean_y) ** 2 for pred_value in pred_y])
        mean_deviation = sum([(true_value - mean_y) ** 2 for true_value in y])

        return pred_deviation / mean_deviation