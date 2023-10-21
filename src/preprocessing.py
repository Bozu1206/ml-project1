import numpy as np
import csv
import polynomial_exp as exp


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
    print(ids)
    assert 0
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


def undefined_to_most_frequent(feature):
    # Exclure les nan et trouver les valeurs uniques et leurs compte
    valeurs_uniques, comptes = np.unique(
        feature[~np.isnan(feature)], return_counts=True
    )

    # Trouver la valeur avec la plus grande fréquence
    most_frequent_value = valeurs_uniques[np.argmax(comptes)]

    # Remplacer les nan par la valeur la plus fréquente
    feature[np.isnan(feature)] = most_frequent_value
    return feature


def undefined_to_avg(arr):
    # Calculer la moyenne en ignorant les nan
    moyenne = np.nanmean(arr)
    # Remplacer les nan par la moyenne
    arr[np.isnan(arr)] = moyenne
    return arr


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def _clean_data_core(feature, to_eleminate=None, to_replace=None):
    if to_eleminate is not None:
        for v in to_eleminate:
            feature[feature == v] = np.nan
    if to_replace is not None:
        for key, value in to_replace.items():
            feature[feature == key] = value
    return feature


def clean_data(features: dict, data_x, do_poly=False):
    headers = []
    with open("../data/raw/x_train.csv", "r") as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames

    assert len(headers) == 322

    # Parse dict
    indices = []
    for col_name, cleaning in features.items():
        to_clean, categorie = cleaning
        values_to_nan = to_clean[0]
        values_to_change = to_clean[1]

        index = headers.index(col_name)
        indices.append(index)
        data_x[:, index] = _clean_data_core(
            data_x[:, index], values_to_nan, values_to_change
        )

        if categorie.find("CON") != -1:
            # Continous feature
            data_x[:, index] = undefined_to_avg(data_x[:, index])
            # Standardize
            data_x[:, index] = standardize(data_x[:, index])

            if categorie.find("Poly") != -1:
                # Do polynomial expansion here, if needed
                pass
                # data_x = exp.compute_and_add_poly_expansion(data_x[:, index], data_x, degree=3, f=col_name)

        if categorie.find("CAT") != -1:
            # Categorical feature: replace by the most frequent values
            data_x[:, index] = undefined_to_most_frequent(data_x[:, index])

    data_x = data_x[:, indices]

    if do_poly:
        for index, (col_name, cleaning) in enumerate(features.items()):
            _, categorie = cleaning
            if categorie.find("Poly") != -1:
                # Do polynomial expansion here, if needed
                # pass
                data_x = exp.compute_and_add_poly_expansion(
                    data_x[:, index], data_x, degree=3, f=col_name
                )

    return data_x
