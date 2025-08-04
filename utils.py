import numpy as np
import copy

def remove_duplicate_data(data):
    _data = copy.deepcopy(data)
    sort_idx = _data[:, -1].argsort()
    sort_data = _data[sort_idx]
    _, uni_idx = np.unique(sort_data, axis=0, return_index=True)
    uni_data = sort_data[uni_idx]
    return uni_data