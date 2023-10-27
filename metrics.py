import numpy as np
from colors import style


def compute_accuracy(y_true, y_pred):
    matches = np.sum(y_true == y_pred)
    return matches / y_true.shape[0]


def f1_score(y_true, y_pred):
    assert len(y_true) == len(
        y_pred
    ), "Length of actual and predicted values must be the same"

    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if true == -1 and pred == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == -1)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    return f1


def agg_results_and_print_results(model, y_test, x_test, weights, file_name=None):
    if file_name is None:
        # Print in standard output
        test_preds = model.predict(x_test, weights)
        test_preds[np.where(test_preds == 0)] = -1  # In case of logistic regression
        testing_accuracy = compute_accuracy(y_test, test_preds)
        f1score = f1_score(y_test, test_preds)

        print(
            style.BOLD
            + style.GREEN
            + f"\tTest accuracy: {testing_accuracy:.4f}"
            + style.RESET
        )
        print(style.BOLD + style.YELLOW + f"\tF1-Score: {f1score:.4f}" + style.RESET)
