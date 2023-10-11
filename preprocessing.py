import numpy as np

def split_by_category(id, y, x, col):
    """
    Split the dataset according to column value.

    Args:
        id: shape=(N, )
        y: shape=(N, )
        x: shape=(N, D)
        col: int, the feature identifier
    """

    data = np.c_[id, y, x]
    split = [data[data[:, col+2] == k] for k in np.unique(data[:, col+2])]
    ids = [mat[:, 0] for mat in split]
    ys = [mat[:, 1] for mat in split]
    # delete the category column
    xs = [np.delete(mat[:, 2:], 22, 1) for mat in split]

    return ids, ys, xs

def prune_undefined_features(x, undefined=np.nan):
    """
    Delete all undefined features of variance zero.
    """

    idx = np.argwhere(np.all(x[..., :] == undefined, axis=0))
    return np.delete(x, idx, axis=1)


def impute_undefined_values(x, undefined=np.nan):
    """
    Replaces undefined values by the feature average.
    """
    col_mean = np.mean(x, where=(x!=undefined), axis=0)
    mask = np.where(x==undefined)
    x[mask] = np.take(col_mean, mask[1])

    return x