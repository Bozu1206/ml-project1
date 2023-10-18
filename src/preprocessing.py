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

def change_x_by_nan(feature, to_eleminate, to_replace=None):
    for v in to_eleminate: 
        feature[feature == v] = np.nan
    if to_replace is not None:
        for key, value in to_replace.items():
            feature[feature == key] = value
    return feature

def clean_features(tx):
    indices = [26,27,28,30,32,33,39,42,43,44,46,48,50,80,232,234,237,238,242,246,253,259,264,278,279,284,288]
    #indices = [27,28,29,31,33,34,40,43,44,45,47,49,51,81,233,235,238,239,243,247,254,260,265,279,280,285,289]
    x = tx[:, indices]
    change_x_by_nan(x[:, 0], [7, 9, 88]) # GENHEALT
    change_x_by_nan(x[:, 1], [77, 99], {88: 0}) # PHYSHEAT
    change_x_by_nan(x[:, 2], [77, 99], {88: 0}) # MENTALHEALTH
    change_x_by_nan(x[:, 3],  [7, 9], {2: 0}) # HLTHPLN1
    change_x_by_nan(x[:, 4],  [7, 9], {2: 0}) # MEDCOST1
    change_x_by_nan(x[:, 5],  [7, 9], {8: 5}) # CHECKUP1
    change_x_by_nan(x[:, 6],  [7, 9], {2: 0}) # CVDSTRK3
    change_x_by_nan(x[:, 7],  [7, 9], {2: 0}) # CHCSCNCR
    change_x_by_nan(x[:, 8],  [7, 9], {2: 0}) # CHCOCNCR
    change_x_by_nan(x[:, 9],  [7, 9], {2: 0}) # CHCCOPD1
    change_x_by_nan(x[:, 10], [7, 9], {2: 0}) # ADDEPEV2
    change_x_by_nan(x[:, 23], [9], {2:0}) # _FRTLT1 
    change_x_by_nan(x[:, 24], [9], {2:0}) # _VEGLT1 
    change_x_by_nan(x[:, 25], [9], {2:0}) # _TOTINDA 
    change_x_by_nan(x[:, 11],  [7, 9], {2:0, 3:0, 1:2, 4:1}) # DIABETE3 ASK ELISA
    #change_x_by_nan(x[:, 12], [7, 9], {2: 0}) # SEX ASK ELISA (Nothing to change)
    change_x_by_nan(x[:, 13], [7, 9]) # MAXDRNKS
    change_x_by_nan(x[:, 14], [9], {1:0, 2:1}) # _RFHYPE5
    change_x_by_nan(x[:, 15], [9], {1:0, 2:1}) # _RFCHOL
    change_x_by_nan(x[:, 16], [9], {3:0, 2:1, 1:2}) # _ASTHMS1 
    change_x_by_nan(x[:, 17], [], {2:0}) # _DRDXAR1 
    change_x_by_nan(x[:, 19], [14]) # _AGEG5YR 
    change_x_by_nan(x[:, 20], []) # _BMI5 
    change_x_by_nan(x[:, 18], [9]) # _RACE 
    change_x_by_nan(x[:, 21], [9]) # _SMOKER3 
    change_x_by_nan(x[:, 22], [99900]) # _DRNKWEK 
    change_x_by_nan(x[:, 26], [99900]) # FC60_ 
    return x