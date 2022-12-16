import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from ..core import SearchBase
from ..utils import hasarg


def compute_class_weights(y, return_dict=False):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    if return_dict:
        class_weights = dict(zip(classes, weights))
        return class_weights
    return weights


def _get_grid(grids_dict, grid_mode):
    modes = ['light', 'medium', 'hardcore']

    if not isinstance(grids_dict, dict):
        raise TypeError("'grids_dict' must be a dictionary")

    at_least_one = False
    for mode in modes:
        if mode in grids_dict:
            at_least_one = True
            globals()[f'grid_{mode}'] = grids_dict[mode]
        else:
            globals()[f'grid_{mode}'] = {}

    if not at_least_one:
        raise ValueError(f'At least one grid-mode form {tuple(modes)} must be specified')

    def _construct_grid():
        modes = iter(['light', 'medium', 'hardcore'])
        grids = iter([grid_light, grid_medium, grid_hardcore, {}])

        def cursor(base_grid, add_grid, mode, target_mode=grid_mode):
            if mode == target_mode:
                return base_grid
            return cursor({**base_grid, **add_grid}, next(grids), next(modes))

        out = cursor(next(grids), next(grids), next(modes))

        if not out:
            raise Exception(f"grid_mode '{grid_mode}' is not specified and does not have a backup")

        return out

    if isinstance(grid_mode, str):
        try:
            return _construct_grid()
        except StopIteration:
            raise ValueError(f"Invalid grid_mode '{grid_mode}'")

    return grid_mode


def aliases(name):
    groups = (
              ('cl', 'classifier', 'classification'),
              ('reg', 'regressor', 'regression'),
              ('n_jobs', 'num_cores', 'thread_count', 'n_cores'),
              ('verbose', 'verbosity'),
              ('random_state', 'seed', 'random_seed')
              )

    for group in groups:
        if name in group:
            return group

    return ()


def _get_base(task, n_jobs, verbose, classifier, regressor, base_params=None):
    def _get_estimator(specs):
        if isinstance(specs, dict):
            if 'estimator' in specs:
                estimator = specs['estimator']
                if 'const_params' in specs:
                    const_params = specs['const_params']
                    return estimator, const_params
                return estimator, {}
            raise KeyError("key 'estimator' not found in classifier")
        return specs, {}

    if task in aliases('cl'):
        estimator, const_params = _get_estimator(classifier)
    elif task in aliases('reg'):
        estimator, const_params = _get_estimator(regressor)
    else:
        raise ValueError('Wrong task type')

    if base_params is None:
        base_params = {}

    for alias in aliases('n_jobs'):
        if hasarg(estimator, alias):
            base_params[alias] = n_jobs
            break

    for alias in aliases('verbose'):
        if hasarg(estimator, alias):
            base_params[alias] = verbose
            break

    for alias in aliases('random_state'):
        if hasarg(estimator, alias):
            base_params[alias] = np.random.randint(0, 1e+5)
            break

    return estimator(), const_params, base_params


def make_searcher(grids, classifier, regressor):
    class searcher(SearchBase):
        GRIDS = grids
        CLASSIFIER = classifier
        REGRESSOR = regressor

        @classmethod
        def _grid(cls, grid_mode, shape=None):

            return _get_grid(cls.GRIDS, grid_mode)

        @classmethod
        def _estimator_base(cls, task, n_jobs, verbosity):

            return _get_base(task, n_jobs, verbosity, cls.CLASSIFIER, cls.REGRESSOR)

    return searcher
