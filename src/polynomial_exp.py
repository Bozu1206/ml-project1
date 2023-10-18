import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    assert degree > 0

    poly = x
    for deg in range(2, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    one = np.ones([poly.shape[0], 1])
    return np.c_[one, poly]


def add_poly_expansion_on_data_matrix(poly_exp, tX):
    return np.concatenate((tX, poly_exp), axis=1)


def compute_and_add_poly_expansion(feature, tX, degree=3):
    return add_poly_expansion_on_data_matrix(build_poly(feature, degree), tX)

def build_poly(tx, degree, do_add_bias=True, odd_only=False):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    _, D = tx.shape
    new_tx = np.zeros((tx.shape[0], degree * D))

    step = 2 if odd_only else 1

    j = 0
    for feat in range(0, D):
        for i in range(1, degree + 1, step):
            new_tx[:, j] = np.power(tx[:, feat], i)
            j = j + 1

    return np.concatenate((np.ones((tx.shape[0], 1)), new_tx), axis=1) if do_add_bias else new_tx
