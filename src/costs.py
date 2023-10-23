import numpy as np


def mse(error):
    return 0.5 * np.mean(error**2)


def mae(error):
    return np.mean(np.abs(error))


def compute_mse(y, tx, w):
    error = y - tx.dot(w)
    return mse(error)


def compute_rmse(y, tx, w):
    error = y - tx.dot(w)
    return np.sqrt(2 * mse(error))


def compute_mae(y, tx, w):
    error = y - tx.dot(w)
    return mae(error)


def sigmoid(z):
    z = np.clip(z, -1e5, 1e5)
    return 1.0 / (1 + np.exp(-z))


def compute_log_loss(y, tx, w):
    sig = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(sig) + (1 - y) * np.log(1 - sig))
    return loss


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y:  shape=(N, )
        tx: shape=(N,2)
        w:  shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return compute_mse(y, tx, w)
