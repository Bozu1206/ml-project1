import numpy as np
import csv
import src.polynomial_exp as exp
import random


def balance_data(x, y, seed, size):
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y != 1)[0]

    min_samples = min(len(positive_indices), len(negative_indices))

    prop_maj_class = int(size * float(min_samples))

    random.seed(seed)
    downsampled_negative_indices = random.sample(list(negative_indices), prop_maj_class)

    balanced_indices = np.concatenate([positive_indices, downsampled_negative_indices])
    random.shuffle(balanced_indices)

    balanced_x = x[balanced_indices]
    balanced_y = y[balanced_indices]

    print(
        f"Training with : {prop_maj_class / len(balanced_x) * 100:.2f}% of [-1] and with {len(positive_indices) / len(balanced_x) * 100:.2f}% of [1]"
    )

    return balanced_x, balanced_y


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


def undefined_to_median(x, undefined=np.nan):
    """
    Replaces undefined values by the feature median.
    """
    median = np.nanmedian(x)
    x[np.isnan(x)] = median
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

def one_hot_encode(x):
    """
    Returns a One-hot encoding of a categorical feature
    """
    labels = np.unique(x)
    one_hot = np.zeros((len(x), len(labels)), dtype=int)

    for i, label in enumerate(labels):
        one_hot[np.where(x == label), i] = 1

    return one_hot

def _clean_data_core(feature, to_eleminate=None, to_replace=None):
    if to_eleminate is not None:
        for v in to_eleminate:
            feature[feature == v] = np.nan
    if to_replace is not None:
        for key, value in to_replace.items():
            feature[feature == key] = value
    return feature


def clean_data(features: dict, data_x, median_estimator=False, do_poly=False, do_one_hot=False):
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
            if median_estimator:
                data_x[:, index] = undefined_to_median(data_x[:, index])
            else:
                data_x[:, index] = undefined_to_avg(data_x[:, index])
            # Standardize
            data_x[:, index] = standardize(data_x[:, index])

        if categorie.find("CAT") != -1:
            # Categorical feature: replace by the most frequent values
            data_x[:, index] = undefined_to_most_frequent(data_x[:, index])

    if do_one_hot and not do_poly:
        new_data = np.empty((data_x.shape[0]))

        for col, cleaning in features.items():
            _, label = cleaning 
            index = headers.index(col)

            if index in indices and label.find("CAT") != -1:
                new_data = np.c_[new_data, one_hot_encode(data_x[:, index])]
            else:
                new_data = np.c_[new_data, data_x[:, index]]

        return new_data

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
