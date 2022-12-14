import numpy as np
from scipy.stats import uniform, randint
from sklearn.svm import SVC, SVR

from .utils import _construct_grid
from ..core.base import SearchBase


class SVM(SearchBase):
    """Parameter search for SVM and SVC"""
    __doc__ += SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape=None):

        grid_l = dict(C=uniform(0, 100), kernel=['rbf', 'poly'],
                      degree=randint(1, 10), coef0=uniform(0, 10))
        grid_m = dict(C=uniform(0, 10000), degree=randint(1, 100),  gamma=['scale', 'auto'],
                      kernel=['linear', 'poly', 'rbf', 'sigmoid'])
        grid_h = dict(C=uniform(0, 1e+5), gamma=uniform(0,  1))

        if isinstance(grid_mode, str):
            try:
                return _construct_grid(grid_l, grid_m, grid_h, grid_mode)
            except StopIteration:
                raise ValueError('Invalid grid mode')
        return grid_mode

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        if task == 'cl':
            estimator = SVC()
            const_params = {'class_weight': 'balanced'}
        elif task == 'reg':
            estimator = SVR()
            const_params = {}
        else:
            raise ValueError('Invalid task type')

        base_params = {'verbosity': verbosity,
                       'random_state': np.random.randint(0, 1e+5)}

        return estimator, base_params, const_params



