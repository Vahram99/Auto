"""from autopt.models import CatBoost
from dataset import x, y  # Data

optimization = CatBoost(task='cl', grid_mode='medium')
optimization.fit(x, y)


from autopt.core import SearchBase
from autopt.models.utils import _get_grid, _get_base


class MySearch(SearchBase):
    @staticmethod
    def _grid(grid_mode, shape=None):

        grid_light = dict("search space for 'light' mode")
        grid_medium = dict("search space for 'medium' mode")
        grid_hardcore = dict("search space for 'hardcore' mode")

        return _get_grid(grid_l, grid_m, grid_h, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = MySearchClassifier
        regressor = MySearchRegressor

        return _get_base(task, n_jobs, verbosity, classifier, regressor)"""

"""
from abc import abstractmethod


class base:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.v = self._abstract_method(a,b)

    @staticmethod
    @abstractmethod
    def _abstract_method(a, b):
        pass


def make_new(c):
    class new(base):
        C = c

        @classmethod
        def _abstract_method(cls, a, b):
            return cls.C + a + b

    return new


new = make_new(10)

n = new(4,5)
print(n.v)

k = {'a':2}
if k:
    print(True)"""

import numpy as np

l = [np.array([1,2,3,4]),np.array([5,6,7,8]),np.array([9,10,11,12])]

print(np.concatenate(l,axis = 0).reshape(3,-1))