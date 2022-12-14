import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y, return_dict=False):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    if return_dict:
        class_weights = dict(zip(classes, weights))
        return class_weights
    return weights


def _construct_grid(grid_l, grid_m, grid_h, grid_mode):
    modes = iter(['light', 'medium', 'hardcore'])
    grids = iter([grid_l, grid_m, grid_h, {}])

    def cursor(base_grid, add_grid, mode, target_mode=grid_mode):
        if mode == target_mode:
            return base_grid
        return cursor({**base_grid, **add_grid}, next(grids), next(modes))

    out = cursor(next(grids), next(grids), next(modes))
    return out

