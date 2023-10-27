import numpy as np
import csv
import polynomial_exp as exp
import random
import helpers
import json_parser


def resample(
    x, y, seed, d_size=0.5, u_size=0, oversample=False, downsample=True, both=False
):
    np.random.seed(seed)

    negative_indices = np.where(y == -1)[0]
    positive_indices = np.where(y == 1)[0]

    if oversample:
        num_samples = int(u_size * len(positive_indices))
        sampled_positive_indices = np.random.choice(
            positive_indices, num_samples, replace=True
        )
        resampled_x = np.concatenate([x[negative_indices], x[sampled_positive_indices]])
        resampled_y = np.concatenate([y[negative_indices], y[sampled_positive_indices]])

    elif downsample:
        num_samples = int(d_size * len(negative_indices))
        sampled_negative_indices = np.random.choice(
            negative_indices, num_samples, replace=False
        )
        resampled_x = np.concatenate([x[sampled_negative_indices], x[positive_indices]])
        resampled_y = np.concatenate([y[sampled_negative_indices], y[positive_indices]])

    elif both:
        num_negative_samples = int(d_size * len(negative_indices))
        num_positive_samples = int(u_size * len(positive_indices))
        sampled_negative_indices = np.random.choice(
            negative_indices, num_negative_samples, replace=False
        )
        sampled_positive_indices = np.random.choice(
            positive_indices, num_positive_samples, replace=True
        )
        resampled_x = np.concatenate(
            [x[sampled_negative_indices], x[sampled_positive_indices]]
        )
        resampled_y = np.concatenate(
            [y[sampled_negative_indices], y[sampled_positive_indices]]
        )

    else:
        raise ValueError()

    return resampled_x, resampled_y


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


def undefined_to_median(x, undefined=np.nan):
    """
    Replaces undefined values by the feature median.
    """
    median = np.nanmedian(x)
    x[np.isnan(x)] = median
    return x


def undefined_to_most_frequent(feature):
    valeurs_uniques, comptes = np.unique(
        feature[~np.isnan(feature)], return_counts=True
    )

    most_frequent_value = valeurs_uniques[np.argmax(comptes)]
    feature[np.isnan(feature)] = most_frequent_value
    return feature


def undefined_to_minus_one(feature):
    feature[np.isnan(feature)] = -1
    return feature


def undefined_to_avg(x):
    mean = np.nanmean(x)
    x[np.isnan(x)] = mean
    return x


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


def clean_data(
    features: dict,
    data_x,
    median_estimator=False,
    do_poly=False,
    do_one_hot=False,
    minus_one=False,
):
    headers = []
    with open("./data/raw/x_train.csv", "r") as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames

    headers = headers[1:]  # Remove id col
    assert len(headers) == 321

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
            if minus_one:
                data_x[:, index] = undefined_to_minus_one(data_x[:, index])
            else:
                # Categorical feature: replace by the most frequent values
                data_x[:, index] = undefined_to_most_frequent(data_x[:, index])

    # Redo this
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

    indices = np.sort(indices)
    data_x = data_x[:, indices]

    if do_poly:
        for index, (col_name, cleaning) in enumerate(features.items()):
            _, categorie = cleaning
            if categorie.find("Poly") != -1:
                # Do polynomial expansion here, if needed
                data_x = exp.compute_and_add_poly_expansion(
                    data_x[:, index], data_x, degree=3, f=col_name
                )

    return data_x


def prepare_data(x, y, seed, ratio, downsampling_size=0, upsamplingsize=0):
    if downsampling_size == 0 and upsamplingsize == 0:
        y_train, x_train, y_test, x_test = helpers.split_data_rand(y, x, ratio)
    else:
        y_train, x_train, y_test, x_test = helpers.split_data_rand(y, x, ratio)

        if downsampling_size != 0 and upsamplingsize == 0:
            # Downsampling
            x_train, y_train = resample(
                x_train, y_train, seed=seed, d_size=downsampling_size, downsample=True
            )
        elif downsampling_size == 0 and upsamplingsize != 0:
            # Upsampling
            x_train, y_train = resample(
                x_train, y_train, seed=seed, u_size=downsampling_size, oversample=True
            )
        elif downsampling_size != 0 and upsamplingsize != 0:
            # Both
            x_train, y_train = resample(
                x_train,
                y_train,
                seed=seed,
                d_size=downsampling_size,
                u_size=downsampling_size,
                both=True,
            )

    # Read features to keep
    features = json_parser.parse_json_file("./features.json")
    x_train = clean_data(
        features,
        x_train,
        do_poly=False,
        minus_one=True,
        median_estimator=True,
        do_one_hot=True,
    )
    x_test = clean_data(
        features,
        x_test,
        do_poly=False,
        minus_one=True,
        median_estimator=True,
        do_one_hot=True,
    )
    return y_train, x_train, y_test, x_test
