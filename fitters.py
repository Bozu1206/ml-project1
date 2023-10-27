import numpy as np
import itertools
import os
import helpers
import implementations as imp
import costs
import polynomial_exp


class GradientFitter:
    ## Here we supposed that data is already cleaned
    def __init__(self, y_train, x_train, y_test, x_test, max_iters, gamma, tresh):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.max_iters = max_iters
        self.gamma = gamma
        self.tresh = tresh

    def __train_and_validate(self):
        # Train a GD Linear regression predictor
        # Kaiming Initialisation
        initial_w = np.random.normal(
            0.0, 2 / self.x_train.shape[1], self.x_train.shape[1]
        )
        w, loss = imp.mean_squared_error_gd(
            self.y_train, self.x_train, initial_w, self.max_iters, self.gamma
        )

        # Test error
        train_loss = costs.compute_mse(self.y_train, self.x_train, w)
        return w, train_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (self.tresh, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred


class StochasticGradientFitter:
    # Here we suppose that data is already cleaned
    def __init__(self, y_train, x_train, y_test, x_test, max_iters, gamma, thresh):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.max_iters = max_iters
        self.gamma = gamma
        self.thresh = thresh

    def __train_and_validate(self):
        # Train a SGD Linear regression predictor
        # Kaiming Initialisation
        initial_w = np.random.normal(
            0.0, 2 / self.x_train.shape[1], self.x_train.shape[1]
        )
        w, loss = imp.mean_squared_error_sgd(
            self.y_train, self.x_train, initial_w, self.max_iters, self.gamma
        )

        # Test error
        train_loss = costs.compute_mse(self.y_train, self.x_train, w)
        return w, train_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (0, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred


class LeastSquareFitter:
    ## Here we supposed that data is already cleaned
    def __init__(self, y_train, x_train, y_test, x_test, thresh):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.thresh = thresh

    def __train_and_validate(self):
        # Train a Least Squares predictor
        w, _ = imp.least_squares(self.y_train, self.x_train)

        # Test error
        test_loss = costs.compute_rmse(self.y_test, self.x_test, w)
        return w, test_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (self.thresh, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred


class RidgeRegressionFitter:
    def __init__(self, y_train, x_train, y_test, x_test, lambda_, thresh):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.lambda_ = lambda_
        self.thresh = thresh

    def __train_and_validate(self):
        # Train a Regularized Least Square predictor
        w, _ = imp.ridge_regression(self.y_train, self.x_train, self.lambda_)

        # Test error
        test_loss = costs.compute_rmse(self.y_test, self.x_test, w)
        return w, test_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (self.thresh, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred


class LogisticRegressionFitter:
    ## Here we supposed that data is already cleaned
    def __init__(self, y_train, x_train, y_test, x_test, max_iters, gamma, thresh):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.max_iters = max_iters
        self.gamma = gamma
        self.thresh = thresh

    def __train_and_validate(self):
        # Kaiming Initialisation
        initial_w = np.random.normal(
            0.0, 2 / self.x_train.shape[1], self.x_train.shape[1]
        )
        w, _ = imp.logistic_regression(
            self.y_train, self.x_train, initial_w, self.max_iters, self.gamma
        )

        # Test error
        test_loss = costs.compute_log_loss(self.y_test, self.x_test, w)
        return w, test_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (self.thresh, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred


class RegLogisticRegressionFitter:
    ## Here we supposed that data is already cleaned
    def __init__(
        self, y_train, x_train, y_test, x_test, max_iters, gamma, lambda_, thresh
    ):
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.max_iters = max_iters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.thresh = thresh

    def __train_and_validate(self):
        # Kaiming Initialisation
        initial_w = np.random.normal(
            0.0, 2 / self.x_train.shape[1], self.x_train.shape[1]
        )
        w, _ = imp.reg_logistic_regression(
            self.y_train,
            self.x_train,
            self.lambda_,
            initial_w,
            self.max_iters,
            self.gamma,
        )
        test_loss = costs.compute_log_loss(self.y_test, self.x_test, w)
        return w, test_loss

    def fit(self):
        return self.__train_and_validate()

    def predict(self, data, weights):
        """Generate class predictions given weights, and a test data matrix."""
        y_pred = data.dot(weights)
        cutoff, lower, upper = (self.thresh, -1, 1)
        y_pred[np.where(y_pred <= cutoff)] = lower
        y_pred[np.where(y_pred > cutoff)] = upper
        return y_pred
