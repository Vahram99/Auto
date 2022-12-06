import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, clone
from .search import BayesianSearchCV


class SearchBase(BaseEstimator, metaclass=ABCMeta):
    """Base class for all optimizers

    Parameters
    ----------
    task : str
       Task type of the estimator

       -'cl'  for classifiers
       -'reg'  for regressors

    scoring : str, callable or dict, default=None
       A single str, callable or a dict
       to evaluate the predictions on the test set.
       Dict template -> {'scoring':callable,'maximize':True},
       active only if search_mode = 'bayesian' or is a BayesianSearchCV instance

    grid_mode : str or dict
        Possible values are

        - 'light'
        - 'medium'
        - 'hardcore'
        -  dictionary with parameters names (`str`) as keys and lists of
       parameter settings to try as value

    search_mode : str or callable
        Possible values are

        - 'random' - invokes sklearn.RandomizedSearchCV
        - 'bayesian' - invokes auto.BayesianSearchCV

    cv : int, cross-validation generator or an iterable, default=5
      Determines the cross-validation splitting strategy.
       Possible inputs for cv are:

       - integer, to specify the number of folds in a `(Stratified)KFold`,
       - :term:`CV splitter`,
       - An iterable yielding (train, test) splits as arrays of indices.

    cv_repeats: int, default=None
       Invokes RepeatedKFold
       Active only if cv is int

    refit: bool, default=True
       Refit an estimator using the best found parameters on the whole dataset

    calibrate: bool, default = False
       Calibrate the best estimator
       active only if task == 'cl'

    search_verbosity : bool, default=False
       verbosity for the parameter search

    model_verbosity : bool, default=False
       verbosity for the training process of the estimator

    n_jobs : int, default=1
       Number of jobs to run in parallel.
       ``-1`` means using all processors.

    pre_dispatch: int or str, default=’2*n_jobs’
       Controls the number of jobs that get dispatched during parallel execution

    const_params : dict, default=None
       Names and values of the parameters out of search
       Dictionary with parameters names (`str`) as keys and lists of
       parameter settings to try as value

    **search_params
       Additional parameters for the lookup

    Attributes
    ----------

    cv_results_: dict of numpy arrays
       A dict with keys as column headers and values as columns,
       that can be imported into a pandas DataFrame

    best_estimator_: estimator
       Best estimator that was chosen by the search, i.e. estimator which gave
       the highest score (or smallest loss if specified) on the left out data.
       Not available if refit=False

    best_score_: float
       Mean cross-validated score of the best_estimator

    best_params_: dict
       Parameter setting that gave the best results on the hold out data

    cc_: estimator
       Best estimator calibrated
       Only available if task == 'cl'
       Not available if refit=False
       Not available if calibrate=False

 """

    def __init__(self, task, scoring=None, grid_mode='light', search_mode='bayesian', cv=5,
                 cv_repeats=None, refit=True, calibrate=False, search_verbosity=False, model_verbosity=False,
                 n_jobs=-1, pre_dispatch="2*n_jobs", const_params=None, **search_params):

        self.task = task
        self.scoring = scoring
        self.grid_mode = grid_mode
        self.search_mode = search_mode
        self.cv = cv
        self.cv_repeats = cv_repeats
        self.refit = refit
        self.calibrate = calibrate
        self.search_verbosity = search_verbosity
        self.model_verbosity = model_verbosity
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.const_params = const_params
        self.search_params = search_params
        self._param_search = self._search(search_mode)

        if cv_repeats and isinstance(cv, int):
            self.cv = RepeatedKFold(n_splits=cv, n_repeats=cv_repeats,
                                    random_state=np.random.randint(0, 1e+5))

        estimator, base_params, const_params_ = self._estimator_base(task, n_jobs=n_jobs,
                                                                     verbosity=model_verbosity)
        if not const_params:
            const_params = const_params_

        estimator.set_params(**base_params)
        estimator.set_params(**const_params)
        self._estimator = estimator

    def fit(self, x, y, **fit_params):
        """
        Run optimization on the search space

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output)
            Target relative to X for classification or regression

        Returns self: object
        """

        param_grid = self._grid(self.grid_mode, x.shape)

        lookup = self._param_search(self._estimator, param_grid, cv=self.cv,scoring=self.scoring,
                                    refit=self.refit, n_jobs=self.n_jobs,pre_dispatch=self.pre_dispatch,
                                    verbose=self.search_verbosity, **self.search_params)

        lookup.fit(x, y, **fit_params)

        self.cv_results_ = lookup.cv_results_
        self.best_params_ = lookup.best_params_
        self.best_score_ = lookup.best_score_
        self.best_estimator_ = lookup.best_estimator_

        if self.refit & self.calibrate & (self.task == 'cl'):
            if x.shape[0] < 1000:
                method = 'sigmoid'
            else:
                method = 'isotonic'

            cc = CalibratedClassifierCV(base_estimator=clone(self._estimator), cv=self.cv,
                                        n_jobs=self.n_jobs, method=method)
            cc.fit(x, y)
            self.cc_ = cc

    def save(self, path_to_file=None):
        """
        Save the results to the file specified

        Parameters
        ----------
        path_to_file : str, default=None
           File path, where the results should be saved

        Returns self: object"""

        attrs = ['cv_results_',
                 'best_params_',
                 'best_score_',
                 'best_estimator_',
                 'cc_']

        present_attrs = set(attrs) & set(self.__dict__)
        results = {attr: getattr(self, attr) for attr in present_attrs}

        if not path_to_file:
            if 'write_path' in self.search_params:
                path_to_file = self.search_params['write_path']
            else:
                raise ValueError('File path is not specified')

        with open(path_to_file, 'wb') as f:
            pickle.dump(results, f)

    @staticmethod
    @abstractmethod
    def _grid(grid_mode, shape):
        """Search parameters need to be specified here for the grid modes
             - 'light'
             - 'medium'
             - 'hardcore'

           Some search parameters depend on the number of instances,
           shape param is the shape of the data """
        pass

    @staticmethod
    @abstractmethod
    def _estimator_base(task, n_jobs, verbosity):
        """Returns the estimator with its base and constant parameters"""
        pass

    @staticmethod
    def _search(search_mode):
        if search_mode:
            if search_mode == 'bayesian':
                return BayesianSearchCV
            elif search_mode == 'random':
                return RandomizedSearchCV
            else:
                raise ValueError('Invalid search optimization')
        return None
