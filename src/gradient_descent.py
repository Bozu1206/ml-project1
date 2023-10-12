import numpy as np
from costs import compute_loss


def mse_gradient(y, tx, w):
    error = y - tx.dot(w)
    return -tx.T.dot(error) / len(error)


def mae_gradient(y, tx, w):
    e = y - tx.dot(w)
    return (-1 / len(e)) * tx.T.dot(np.sign(e))


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return mse_gradient(y, tx, w)
