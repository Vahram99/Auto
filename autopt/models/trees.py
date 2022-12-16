from math import ceil
import numpy as np
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from .utils import compute_class_weights, _get_grid, _get_base
from ..core.base import SearchBase


class RandomForest(SearchBase):
    """Parameter search for RandomForestClassifier and RandomForestRegressor"""
    __doc__ += SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape=None):

        grid_l = dict(n_estimators=randint(10, 200), max_depth=randint(3, 10),
                      min_samples_split=uniform(0, 0.1), min_samples_leaf=uniform(0, 0.1))
        grid_m = dict(n_estimators=randint(10, 500), max_depth=randint(3, 15),
                      criterion=['gini', 'entropy', 'log_loss'], max_features=['sqrt', 'log2', None],
                      min_weight_fraction_leaf=uniform(0, 0.1))
        grid_h = dict(max_samples=uniform(0.5, 0.5))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': RandomForestClassifier,
                      'const_params': {'class_weight': 'balanced'}
                      }
        regressor = {'estimator': RandomForestRegressor,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)


class ExtraTrees(SearchBase):
    """Parameter search for ExtraTreesClassifier and ExtraTreesRegressor"""
    __doc__ += SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape=None):

        grid_l = dict(n_estimators=randint(10, 200), max_depth=randint(3, 10),
                      min_samples_split=uniform(0, 0.1), min_samples_leaf=uniform(0, 0.1))
        grid_m = dict(n_estimators=randint(10, 500), max_depth=randint(3, 15),
                      criterion=['gini', 'entropy', 'log_loss'], max_features=['sqrt', 'log2', None],
                      min_weight_fraction_leaf=uniform(0, 0.1))
        grid_h = dict(max_samples=uniform(0.5, 0.5))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': ExtraTreesClassifier,
                      'const_params': {'class_weight': 'balanced'}
                      }
        regressor = {'estimator': ExtraTreesRegressor,
                     'const_params': {}
                     }

        base_params = {'bootstrap': True}

        return _get_base(task, n_jobs, verbosity, classifier, regressor, base_params)


class XGBoost(SearchBase):
    """Parameter search for XGBClassifier and XGBRegressor"""
    __doc__ += '\n'*2 + SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape=None):

        grid_l = dict(n_estimators=randint(10, 100), max_depth=randint(3, 10),
                      learning_rate=uniform(1e-5, 0.3), min_child_weight=uniform(1, 10))
        grid_m = dict(n_estimators=randint(10, 1000), max_depth=randint(3, 15),
                      reg_alpha=uniform(0, 10), reg_lambda=uniform(0, 10),
                      base_score=uniform(0.1, 0.9), colsample_bytree=uniform(0.1, 0.9),
                      colsample_bylevel=uniform(0.1, 0.9), colsample_bynode=uniform(0.1, 0.9))
        grid_h = dict(gamma=randint(0, 10), grow_policy=[0, 1],
                      booster=['gbtree', 'gblinear', 'dart'])

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': LGBMClassifier,
                      'const_params': {}
                      }
        regressor = {'estimator': LGBMRegressor,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)

    def fit(self, x, y, **fit_params):
        class_weight = compute_class_weights(y)
        self._estimator.set_params(**{'scale_pos_weight': max(class_weight)/min(class_weight)})

        super().fit(x, y, **fit_params)


class CatBoost(SearchBase):
    """Parameter search for CatBoostClassifier and CatBoostRegressor"""
    __doc__ += '\n'*2 + SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape):
        n_inst = shape[0]

        grid_l = dict(n_estimators=randint(10, 100), max_depth=randint(3, 10),
                      learning_rate=uniform(1e-5, 0.3),
                      min_data_in_leaf=randint(2, ceil(n_inst/10)))
        grid_m = dict(n_estimators=randint(10, 1000), max_depth=randint(3, 12),
                      random_strength=uniform(1, 10), l2_leaf_reg=uniform(0, 10),
                      rsm=uniform(0, 1))
        grid_h = dict(bagging_temperature=uniform(0, 10),
                      grow_policy=['SymmetricTree', 'Depthwise', 'Lossguide'],
                      leaf_estimation_iterations=randint(1, 10))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': CatBoostClassifier,
                      'const_params': {'auto_class_weights': 'Balanced'}
                      }
        regressor = {'estimator': CatBoostRegressor,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)


class HistGradientBoosting(SearchBase):
    """Parameter search for HistGradientBoostingClassifier and HistGradientBoostingRegressor"""
    __doc__ += '\n'*2 + SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape):
        n_inst = shape[0]

        grid_l = dict(max_iter=randint(10, 100), max_depth=randint(3, 10),
                      learning_rate=uniform(1e-5, 0.3), min_samples_leaf=randint(2, ceil(n_inst/10)))
        grid_m = dict(max_iter=randint(10, 1000), max_depth=randint(3, 15),
                      l2_regularization=uniform(0, 100))
        grid_h = dict(max_leaf_nodes=randint(10, 100))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': HistGradientBoostingClassifier,
                      'const_params': {'class_weight': 'balanced'}
                      }
        regressor = {'estimator': HistGradientBoostingRegressor,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)


class LGBM(SearchBase):
    """Parameter search for LGBMClassifier and LGBMRegressor"""
    __doc__ += '\n'*2 + SearchBase.__doc__.split('\n', 2)[2]

    @staticmethod
    def _grid(grid_mode, shape):
        n_inst = shape[0]

        grid_l = dict(n_estimators=randint(10, 100), max_depth=randint(3, 10),
                      learning_rate=uniform(1e-5, 0.3), min_child_weight=uniform(1, 10),
                      min_child_samples=randint(2, ceil(n_inst/10)))
        grid_m = dict(n_estimators=randint(10, 1000), max_depth=randint(3, 15),
                      min_split_gain=uniform(0, 1), subsample=uniform(0.1, 0.9),
                      colsample_bytree=uniform(0.1, 0.9), colsample_bynode=uniform(0.1, 0.9),
                      reg_alpha=uniform(0, 10), reg_lambda=uniform(0, 10),
                      subsample_freq=randint(1, 10))
        grid_h = dict(drop_rate=uniform(0, 0.3), num_leaves=randint(10, 100))

        grids_dict = dict(light=grid_l, medium=grid_m, hardcore=grid_h)

        return _get_grid(grids_dict, grid_mode)

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = {'estimator': LGBMClassifier,
                      'const_params': {'class_weight': 'balanced'}
                      }
        regressor = {'estimator': LGBMRegressor,
                     'const_params': {}
                     }

        return _get_base(task, n_jobs, verbosity, classifier, regressor)
