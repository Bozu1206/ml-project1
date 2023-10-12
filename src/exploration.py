"""
Helper methods to explore datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def display_parallel_coordinates(y, x, features, colors, labels):
    """
    Displays the parallel coordinates of the input data, useful for data with more than 4 dimensions.

    Args:
        y: shape=(C,)
        x: shape=(N, D)
        colors: shape=(C,)
        labels: shape=(C,)
    """

    fig, axes = plt.subplots(1, features.shape[0] - 1, sharey=False, figsize=(30, 8))
    x_norm = np.zeros_like(x)

    min_max_range = {}
    for i, feature in enumerate(features):
        min_max_range[feature] = [x[:, i].min(), x[:, i].max(), np.ptp(x[:, i])]
        x_norm[:, i] = np.true_divide(x[:, i] - x[:, i].min(), np.ptp(x[:, i]))

    for i in range(len(axes)):
        for j in range(x.shape[0]):
            axes[i].plot(features, x_norm[j], colors[y[j]], alpha=0.7)
        axes[i].set_xlim([features[i], features[i+1]])

    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[features[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = x_norm[:, dim].min()
        norm_range = np.ptp(x_norm[:, dim])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=4)
        ax.set_xticklabels([features[dim]])

    plt.subplots_adjust(wspace=0)

    plt.legend(labels)

    plt.show()