import numpy as np
import implementations as imp
import metrics
import fitters
import preprocessing
import helpers


class CValidator:
    def __init__(self, y, x, K, parameters, model, costf, name, double):
        self.y = y
        self.x = x
        self.K = K
        self.parameters = parameters
        self.model = model
        self.costf = costf
        self.name = name
        self.double = double

    def __create_k_indices(self, y, k_fold, seed=1):
        """build k indices for k-fold.

        Args:
            y:      shape=(N,)
            k_fold: K in K-fold, i.e. the fold num
            seed:   the random seed

        Returns:
            A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

        >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
        array([[3, 2],
            [0, 1]])
        """

        # Set the seed
        np.random.seed(seed)
        interval = int(y.shape[0] / k_fold)
        indices = np.random.permutation(y.shape[0])
        k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
        return np.array(k_indices)

    def __cross_validation(self, y, x, k_indices, k, param, X=None, Y=None):
        """return the loss of ridge regression for a fold corresponding to k_indices

        Args:
            y:          shape=(N,)
            x:          shape=(N,)
            k_indices:  2D array returned by build_k_indices()
            k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
            lambda_:    scalar, cf. ridge_regression()
            degree:     scalar, cf. build_poly()

        Returns:
            train and test root mean square errors rmse = sqrt(2 mse)

        >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
        (0.019866645527597114, 0.33555914361295175)
        """

        test_indices = k_indices[k]
        train_indices = k_indices[np.arange(len(k_indices)) != k].flatten()

        y_train = y[train_indices]
        x_train = x[train_indices]

        y_test = y[test_indices]
        x_test = x[test_indices]

        if self.double == True:
            self.model[1] = fitters.RidgeRegressionFitter(
                y_train, x_train, y_test, x_test, param[1], param[0]
            )

        if self.model[0] == "lrgd" and self.name == "gamma":
            self.model[1] = fitters.GradientFitter(
                y_train, x_train, y_test, x_test, 1000, param, 0
            )

        elif self.model[0] == "lrgd" and self.name == "thresh":
            self.model[1] = fitters.GradientFitter(
                y_train, x_train, y_test, x_test, 1000, 0.01, param
            )

        elif self.model[0] == "lrsgd" and self.name == "gamma":
            self.model[1] = fitters.StochasticGradientFitter(
                y_train, x_train, y_test, x_test, 10000, param, 0
            )

        elif self.model[0] == "lrsgd" and self.name == "thresh":
            self.model[1] = fitters.StochasticGradientFitter(
                y_train, x_train, y_test, x_test, 10000, 0.001, param
            )

        elif self.model[0] == "lsq" and self.name == "thresh":
            self.model[1] = fitters.LeastSquareFitter(
                y_train, x_train, y_test, x_test, param
            )

        elif self.model[0] == "rrf" and self.name == "lambda" and not self.double:
            self.model[1] = fitters.RidgeRegressionFitter(
                y_train, x_train, y_test, x_test, param, 0
            )

        elif self.model[0] == "rrf" and self.name == "thresh" and not self.double:
            self.model[1] = fitters.RidgeRegressionFitter(
                y_train, x_train, y_test, x_test, 10e-7, param
            )

        elif self.model[0] == "lg" and self.name == "gamma":
            self.model[1] = fitters.LogisticRegressionFitter(
                y_train, x_train, y_test, x_test, 1000, param, 0.5
            )

        elif self.model[0] == "lg" and self.name == "thresh":
            self.model[1] = fitters.LogisticRegressionFitter(
                y_train, x_train, y_test, x_test, 1000, 0.005, param
            )

        elif self.model[0] == "rlg" and self.name == "gamma":
            self.model[1] = fitters.RegLogisticRegressionFitter(
                y_train, x_train, y_test, x_test, 1000, param, 10e-7, 0.5
            )

        elif self.model[0] == "rlg" and self.name == "thresh":
            self.model[1] = fitters.RegLogisticRegressionFitter(
                y_train, x_train, y_test, x_test, 1000, 0.005, 0.005, 10e-7, param
            )

        elif self.model[0] == "lg" and self.name == "lambda":
            self.model[1] = fitters.RegLogisticRegressionFitter(
                y_train, x_train, y_test, x_test, 1000, 0.005, 0.005, param, 0.5
            )

        w, loss_tr = self.model[1].fit()
        test_preds = self.model[1].predict(x_test, w)
        true_test_preds = self.model[1].predict(X, w)

        test_preds[np.where(test_preds == 0)] = -1  # In case of logistic regression
        true_test_preds[
            np.where(test_preds == 0)
        ] = -1  # In case of logistic regression

        f1score_tr = metrics.f1_score(y_test, test_preds)
        f1score_te = metrics.f1_score(Y, true_test_preds)

        return f1score_tr, f1score_te

    def cross_validates(self, ratio, seed, ds_factor):
        f1_test = []
        y_train, x_train, y_test, x_test = preprocessing.prepare_data(
            self.x, self.y, seed, ratio, ds_factor
        )
        k_indices = self.__create_k_indices(y_train, self.K)

        if self.double and self.model[0] == "rrf":
            # Enable double cross validation for rrf
            for param1 in self.parameters[0]:
                for param2 in self.parameters[1]:
                    for k in range(self.K):
                        _, f1score = self.__cross_validation(
                            y_train,
                            x_train,
                            k_indices,
                            k,
                            (param1, param2),
                            X=x_test,
                            Y=y_test,
                        )
                        f1_test.append((f1score, param1, param2))

            f1_test = sorted(f1_test, key=lambda x: x[0])
            m = f1_test[-1]
            return m[1], m[2], m[0]

        for param in self.parameters:
            temp_f1_te = []
            for k in range(self.K):
                _, f1score = self.__cross_validation(
                    y_train, x_train, k_indices, k, param, X=x_test, Y=y_test
                )

                temp_f1_te.append(f1score)
            f1_test.append(np.max(temp_f1_te))

        best_param = self.parameters[np.argmax(f1_test)]
        best_f1_score = np.max(f1_test)
        return best_param, best_f1_score
