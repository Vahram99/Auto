import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def disable(method):
    def inner(self, *args, **kwargs):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, method.__name__))
    return inner


def compute_class_weights(y, return_dict=False):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    if return_dict:
        class_weights = dict(zip(classes, weights))
        return class_weights
    return weights
