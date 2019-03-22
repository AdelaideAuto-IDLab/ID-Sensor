import numpy as np

from .api import Api
from .dataset import DataSet, TestSet


def combine_prediction_overlap(data, window_size, window_overlap):
    if window_overlap == 1:
        return np.reshape(data, [-1, data.shape[-1]])

    windows = np.reshape(data, [-1, window_size, data.shape[-1]])
    first = windows[::window_overlap]
    second = windows[1:-1:window_overlap]  # Only 1 window overlap is supported for now

    combined = 0.5 * (first + np.concatenate([first[0:1], second], axis=0))
    return np.reshape(combined, [-1, data.shape[-1]])
