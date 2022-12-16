import numpy as np
from scipy.stats import uniform, randint
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .utils import _get_grid, _get_base
from ..core.base import SearchBase


class MLP(SearchBase):
    """Parameter search for MLPClassifier and MLPRegressor"""
    __doc__ += SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape=None):

        hidden_layer_sizes_l = []
        hidden_layer_sizes_m = []
        for i in range(1000):
            hidden_layer_sizes_l.append(np.random.randint(1, 100, np.random.randint(1, 4)))
            hidden_layer_sizes_m.append(np.random.randint(1, 200, np.random.randint(1, 10)))

        grid_l = dict(hidden_layer_sizes=hidden_layer_sizes_l, alpha=uniform(1e-5, 1),
                      activation=['logistic', 'tanh', 'relu'], max_iter=randint(10, 100))
        grid_m = dict(hidden_layer_sizes=hidden_layer_sizes_m, solver=['lbfgs', 'sgd', 'adam'],
                      learning_rate=['constant', 'invscaling', 'adaptive'], learning_rate_init=uniform(1e-5, 0.1),
                      max_iter=randint(10, 500))
        grid_h = dict(power_t=uniform(1e-5, 1), max_iter=randint(10, 1000))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):

        classifier = {'estimator': MLPClassifier,
                      'const_params': {'early_stopping': True}
                      }
        regressor = {'estimator': MLPRegressor,
                     'const_params': {'early_stopping': True}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)






