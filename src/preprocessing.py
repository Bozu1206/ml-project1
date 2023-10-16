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
    split = [data[data[:, col + 2] == k] for k in np.unique(data[:, col + 2])]
    ids = [mat[:, 0] for mat in split]
    ys = [mat[:, 1] for mat in split]
    # delete the category column
    xs = [np.delete(mat[:, 2:], 22, 1) for mat in split]
    return ids, ys, xs


def prune_undefined(x, axis=1, undefined=np.nan):
    """
    Delete all undefined samples if axis = 0, features if axis = 1
    """

    idx = np.argwhere(np.all(x[..., :] == undefined, axis=axis))
    return np.delete(x, idx, axis=axis)


def undefined_to_mean(x, undefined=np.nan):
    """
    Replaces undefined values by the feature average.
    """
    mean = np.nanmean(x, axis=0)
    ids = np.where(np.isnan(x))
    x[ids] = np.take(mean, ids[1])
    return x

def undefined_to_median(x, undefined=np.nan):
    """
    Replaces undefined values by the feature median.
    """
    median = np.nanmedian(x, axis=0)
    ids = np.where(np.isnan(x))
    x[ids] = np.take(median, ids[1])
    return x