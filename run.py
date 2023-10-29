import metrics
import preprocessing
import fitters
import helpers
import colors
import cross_validation as cv
import numpy as np
import time
import costs
import argparse


def main():
    DATAPATH = "./data/raw/"
    SPLIT_RATIO = 0.69
    DOWNSAMPLING_FACTOR = 0.1651
    SEED = 499912

    parser = argparse.ArgumentParser(description="Heart Prediction ML Challenge")
    parser.add_argument("-cv", type=bool, default=False, help="Do cross-validation")
    parser.add_argument(
        "--all",
        type=bool,
        default=False,
        help="Run all our models. If not set, just run our best model (Ridge Regression)",
    )
    parser.add_argument(
        "-ds", type=float, default=DOWNSAMPLING_FACTOR, help="Down-sampling factor"
    )
    args = parser.parse_args()

    if args.ds != DOWNSAMPLING_FACTOR:
        DOWNSAMPLING_FACTOR = args.ds

    colors.print_greet()

    # 1. Load data
    colors.print_loading_data()
    start_time = time.time()
    raw_x_train, raw_x_test, raw_y_train, train_ids, test_ids = helpers.load_csv_data(
        DATAPATH
    )
    end_time = time.time()
    y_train, x_train, y_test, x_test = preprocessing.prepare_data(
        raw_x_train, raw_y_train, SEED, SPLIT_RATIO, DOWNSAMPLING_FACTOR
    )
    colors.print_finish_loading_data(end_time - start_time)

    # 2. Cross-validate parameters
    if args.cv:
        K = 4
        colors.print_cv_lauching_g()
        colors.print_cv_lauching("threshold, lambda", "Regularized Least Square")
        rrf_best_params = {}
        rrf_tresholds = np.linspace(0, 0.2, 20)
        rrf_lambdas = np.linspace(10e-6, 10e-8, 10)
        cvv = cv.CValidator(
            raw_y_train,
            raw_x_train,
            4,
            (rrf_tresholds, rrf_lambdas),
            ["rrf", None],
            costs.compute_rmse,
            "thresh",
            True,
        )
        best_tresh, best_lambda, f1_score = cvv.cross_validates(
            SPLIT_RATIO, SEED, DOWNSAMPLING_FACTOR
        )
        colors.print_cv_best_param(best_tresh, f1_score)
        colors.print_cv_best_param(best_lambda, f1_score)
        rrf_best_params["thresh"] = best_tresh
        rrf_best_params["lambda"] = best_lambda
        colors.print_cv_ending()

    # 3. Train and test ML models
    print("Regularized Least Squares")
    rrf = fitters.RidgeRegressionFitter(y_train, x_train, y_test, x_test, 10e-7, 0.151)
    w_rrf, loss = rrf.fit()
    metrics.agg_results_and_print_results(rrf, y_test, x_test, w_rrf)

    if args.all:
        colors.print_start_best_models()
        print("Linear Regression with Gradient Descent")
        gm = fitters.GradientFitter(y_train, x_train, y_test, x_test, 1000, 0.01, 0)
        w_gm, loss = gm.fit()
        metrics.agg_results_and_print_results(gm, y_test, x_test, w_gm)

        print("Linear Regression with Stochastic Gradient Descent")
        sgm = fitters.StochasticGradientFitter(
            y_train, x_train, y_test, x_test, 10000, 0.001, 0
        )
        w_sgm, loss = sgm.fit()
        metrics.agg_results_and_print_results(sgm, y_test, x_test, w_sgm)

        print("Least Squares")
        lsq = fitters.LeastSquareFitter(y_train, x_train, y_test, x_test, 0)
        w_lsq, loss = lsq.fit()
        metrics.agg_results_and_print_results(lsq, y_test, x_test, w_lsq)

        # For Logistic Regression
        y_train[np.where(y_train == -1)] = 0

        print("Logistic Regression")
        lg = fitters.LogisticRegressionFitter(
            y_train, x_train, y_test, x_test, 1000, 0.005, 0.5
        )
        w_lg, loss = lg.fit()
        metrics.agg_results_and_print_results(lg, y_test, x_test, w_lg)

        print("Regularized Logistic Regression")
        rlg = fitters.RegLogisticRegressionFitter(
            y_train, x_train, y_test, x_test, 1000, 0.005, 10e-4, 0.5
        )
        w_rlg, loss = rlg.fit()
        metrics.agg_results_and_print_results(rlg, y_test, x_test, w_rlg)
        colors.print_end_best_models()

    # 4. Choose best models and make submissions
    helpers.make_submission(rrf, w_rrf, "sub.csv", test_ids, raw_x_test)
    return


if __name__ == "__main__":
    main()
