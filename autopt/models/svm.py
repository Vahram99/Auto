import numpy as np
from scipy.stats import uniform, randint
from sklearn.svm import SVC, SVR

from .utils import _get_grid, _get_base
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

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': SVC,
                      'const_params': {'class_weight': 'balanced'}
                      }
        regressor = {'estimator': SVR,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)



