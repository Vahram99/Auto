from ..utils import disable

from collections.abc import Mapping, Iterable
from functools import partial
from joblib import logger
from timeit import default_timer as timer

import numpy as np
import pickle

from .validation import cross_validate
from GPyOpt.methods import BayesianOptimization
from sklearn.base import BaseEstimator, clone


class BayesianSearchCV(BayesianOptimization, BaseEstimator):
    """Main class to initialize a Bayesian Optimization method.

    Parameters
    ----------

    estimator : estimator object.
       This is assumed to implement the scikit-learn estimator interface.
       Either estimator needs to provide a ``score`` function,
       or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
       Dictionary with parameters names (`str`) as keys and lists of
       parameter settings to try as values, or a list of dictionaries,
       in which each dictionary stands for a single parameter setting,
       with required keys `name`,`type`, and `domain`

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    cv : int, cross-validation generator or an iterable, default=5
      Determines the cross-validation splitting strategy.
       Possible inputs for cv are:

       - integer, to specify the number of folds in a `(Stratified)KFold`,
       - :term:`CV splitter`,
       - An iterable yielding (train, test) splits as arrays of indices.

    init_trials : int, default=None
       Number of initial points that are collected jointly before
       start running the optimization

    n_iter : int, default=None
       Exploration horizon, or number of iterations
       Active only if max_time=None

    max_time : int, default=None
       Exploration horizon of acquisitions in seconds

    eps : float, default=None
       Minimum distance between two consecutive x's to keep running the model
       Used for early stopping

    refit: bool, default=True
       Refit an estimator using the best found parameters on the whole dataset

    n_jobs : int, default=1
       Number of jobs to run in parallel.
       ``-1`` means using all processors.

    pre_dispatch: int or str, default=’2*n_jobs’
       Controls the number of jobs that get dispatched during parallel execution

    verbose : bool, default=False
       Prints the models and other options during the optimization

    write_path : str, default=None
       File path, where the results should be saved

    n_iters_save : int, default=None
       At each n-th iteration the results are saved to the write_path

    return_predictions : bool, default=False
       Whether to return the predictions of each candidate estimator

    **kwargs : extra parameters
       (see ref:'GPyOpt.methods.BayesianOptimization)
       Extra parameters
       ----------------
       model_type : type of model to use as surrogate, default = 'GP'
       initial_design_type: type of initial design, default = 'random'
       acquisition_type : type of acquisition function to use, default = 'EI'
       acquisition_optimizer_type : type of acquisition optimizer to use, default = 'lbfgs'
       evaluator_type : determines the way the objective is evaluated, default = 'sequential'
       batch_size : size of the batch in which the objective is evaluated, default = 1

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
       Parameter setting that gave the best results on the hold out data"""

    class _Report:
        """A class to keep the track of cv results"""
        def __init__(self, cv, init_trials, return_predictions, verbose, s=100):
            self.iter = 0
            self.t = 0
            self.s = s
            self.init_trials = init_trials
            self.return_predictions = return_predictions
            self.verbose = verbose
            self.best_score_ = None
            self.best_params_ = None

            if not isinstance(cv, int):
                cv = cv.get_n_splits()
            self.cv = cv

            self.mean_fit_time = np.zeros(s)
            self.std_fit_time = np.zeros(s)
            self.mean_score_time = np.zeros(s)
            self.std_score_time = np.zeros(s)
            self.params = np.zeros(s, dtype=object)
            self.test_scores = np.zeros((cv, s))
            self.mean_test_score = np.zeros(s)
            self.std_test_score = np.zeros(s)
            self.predictions = None

        def update(self, params, scores, exec_time):
            if self.return_predictions:
                if self.predictions is None:
                    n_columns = scores['predictions'].shape[0]
                    self.predictions = np.zeros((self.s, n_columns))

            np.put(self.mean_fit_time, self.iter, np.mean(scores['fit_time']))
            np.put(self.std_fit_time, self.iter, np.std(scores['fit_time']))
            np.put(self.mean_score_time, self.iter, np.mean(scores['score_time']))
            np.put(self.std_score_time, self.iter, np.std(scores['score_time']))
            np.put(self.params, self.iter, params)
            np.put(self.mean_test_score, self.iter, np.mean(scores['test_score']))
            np.put(self.std_test_score, self.iter, np.std(scores['test_score']))
            self.test_scores[:, self.iter] = scores['test_score']
            if self.return_predictions:
                self.predictions[self.iter, :] = scores['predictions']

            self.iter += 1
            self.t += exec_time
            if self.iter == self.s - 1:
                self.s = 2*self.s
                new_shape = (self.s,)
                self.mean_fit_time.resize(new_shape)
                self.std_fit_time.resize(new_shape)
                self.mean_score_time.resize(new_shape)
                self.std_score_time.resize(new_shape)
                self.params.resize(new_shape)
                self.mean_test_score.resize(new_shape)
                self.std_test_score.resize(new_shape)
                self.test_scores = np.hstack((self.test_scores,
                                              np.zeros(self.test_scores.shape)))
                if self.return_predictions:
                    self.predictions = np.vstack((self.predictions,
                                                  np.zeros(self.predictions.shape)))

            width = 80

            if self.verbose > 0:
                progress_msg = f"{self.cv}/{self.cv}"
                end_msg = f"[{self.iter}][CV {progress_msg}] END "
                result_msg = ""

                if self.verbose > 1:
                    sorted_keys = sorted(params)
                    params_msg = ", ".join(f"{k}={params[k]}" for k in sorted_keys)

                    progress_msg = f"{self.cv}/{self.cv}"
                    end_msg = f"[{self.iter}][CV {progress_msg}] END "
                    result_msg = params_msg + (";" if params_msg else "")
                    if self.verbose > 2:
                        if isinstance(scores['test_score'], dict):
                            for scorer_name in sorted(scores['test_score']):
                                result_msg += f" {scorer_name}: ("
                                result_msg += f"test={scores['test_score'][scorer_name].mean():.3f})"
                        else:
                            result_msg += ", score="
                            result_msg += f"{scores['test_score'].mean():.3f}"
                result_msg += f" total time={logger.short_format_time(self.t)}"

                # Right align the result_msg
                end_msg += "." * (width - len(end_msg) - len(result_msg))
                end_msg += result_msg
                print(end_msg)

        def stage(self):
            width = 80
            if self.verbose > 0:
                if self.iter == 0:
                    msg = f'{self.init_trials} INITIAL TRIALS'
                    print(msg.center(width, '-'))

                if self.iter == self.init_trials:
                    msg = 'BAYESIAN ACQUISITIONS'
                    print(msg.center(width, '-'))

        def report(self):
            s = self.iter
            cv_results = {'mean_fit_time': np.resize(self.mean_fit_time, s),
                          'std_fit_time': np.resize(self.std_fit_time, s),
                          'mean_score_time': np.resize(self.mean_score_time, s),
                          'std_score_time': np.resize(self.std_score_time, s),
                          'params': np.resize(self.params, s).tolist(),
                          'mean_test_score': np.resize(self.mean_test_score, s),
                          'std_test_score': np.resize(self.std_test_score, s)}

            for cv in range(self.cv):
                cv_results['split{}_test_score'.format(cv)] = np.resize(self.test_scores[cv, :], s)

            if self.return_predictions:
                cv_results['predictions'] = self.predictions[:s, :]

            params, scores = np.resize(self.params, s), np.resize(self.mean_test_score, s)
            best_idx = scores.flatten().argsort()[-1]
            best_params = params[best_idx]
            best_score = scores[best_idx]

            self.best_score_ = best_score
            self.best_params_ = best_params
            return cv_results

    def __init__(self, estimator, param_grid, scoring, cv=5, init_trials=None,
                 n_iter=None, max_time=None, eps=None, refit=True,
                 n_jobs=1, pre_dispatch='2*n_jobs', verbose=False,
                 write_path=None, n_iters_save=None, return_predictions=False, **kwargs):

        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_iter, self.init_trials, self.max_time = self._check_trials(n_iter, init_trials,
                                                                          max_time, len(param_grid))
        self.eps = eps
        self.refit = refit
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.verbose = verbose
        self.write_path = write_path
        self.n_iters_save = n_iters_save
        self.return_predictions = return_predictions
        self.kwargs = kwargs
        self._report = self._Report(cv=cv, verbose=verbose,
                                    init_trials=self.init_trials,
                                    return_predictions=return_predictions)

        self._max_iter = self.n_iter - self.init_trials if self.n_iter else None
        self._domain, self._str_params = self._check_bounds(param_grid,
                                                            n_samples=self._max_iter)

        find_keys = np.vectorize(lambda k: k['name'])
        self._keys = find_keys(self._domain)

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
        """
        estimator = clone(self.estimator)
        loss = partial(self._f, estimator=estimator, x=x, y=y, **fit_params)
        super().__init__(f=loss, domain=self._domain, maximize=True,
                         initial_design_numdata=self.init_trials, **self.kwargs)
        super().run_optimization(max_iter=self._max_iter, max_time=self.max_time,
                                 eps=self.eps if self.eps else -1)

        self._get_results(x, y, **fit_params)

    def save(self, path_to_file=None):
        """
        Writes the results to the file specified

        Parameters
        ----------
        path_to_file: str, default = None
           Alias to write_path """

        if path_to_file:
            self.write_path = path_to_file
        if not self.write_path:
            raise ValueError('Path to file is not specified')

        with open(self.write_path, 'wb') as f:
            pickle.dump(self._results, f)

    def _f(self, params, estimator, x, y, **fit_params):
        current_iter = self._report.iter - self.init_trials
        if self.n_iters_save:
            if (current_iter % self.n_iters_save == 0) & (current_iter > 0):
                self._get_results(x, y, **fit_params)
                self.save()

        feed_params = self._get_feed_params(self._domain, params)
        estimator = clone(estimator)
        estimator.set_params(**feed_params)

        self._report.stage()

        start = timer()
        scores = cross_validate(estimator, x, y, scoring=self.scoring, cv=self.cv,
                                fit_params=fit_params, return_predictions=self.return_predictions,
                                verbose=0, n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        end = timer()
        exec_time = end - start

        self._report.update(feed_params, scores, exec_time)
        score = scores['test_score'].mean()

        return score

    def _get_results(self, x, y, **fit_params):
        self.cv_results_ = self._report.report()
        self.best_params_ = self._report.best_params_
        self.best_score_ = self._report.best_score_
        self.cv_ = self.cv

        results = dict(cv_results_=self.cv_results_,
                       best_params_=self.best_params_,
                       best_score_=self.best_score_,
                       base_estimator_=clone(self.estimator))

        if self.refit:
            best_estimator = clone(self.estimator)
            best_estimator.set_params(**self.best_params_)
            best_estimator.fit(x, y, **fit_params)
            self.best_estimator_ = best_estimator

            results['best_estimator_'] = self.best_estimator_

        self._results = results

    def _get_feed_params(self, bounds, next_set):
        params = {}
        for i in range(len(bounds)):
            param_name = bounds[i]['name']
            picked_value = next_set[0, i]
            if param_name in self._str_params:
                picked_value = self._str_params[param_name][self._check_int(picked_value)]
            params[param_name] = self._check_int(picked_value)
        return params

    @staticmethod
    def _check_trials(n_iter, init_trials, max_time, n_params):
        if not init_trials:
            init_trials = n_params

        if not n_iter:
            n_iter = 5 * n_params

        if max_time:
            return None, init_trials, max_time

        if init_trials >= n_iter:
            raise ValueError(f'Total number of iterations should be higher than the '
                             f'number of initial trials, but {n_iter} < {init_trials}')
        if init_trials < n_params:
            raise ValueError(f'Number of initial trials should be at least'
                             f' equal to the number of search params, but {init_trials} < {n_params}')

        return n_iter, init_trials, max_time

    @staticmethod
    def _check_bounds(candidate, n_samples):
        if not n_samples:
            n_samples = 100

        def param_to_bound(name, value):
            bound = dict(name=name, type='discrete')
            if hasattr(value, 'rvs'):
                bound['domain'] = distr_to_discrete(value)
            elif isinstance(value, Iterable):
                bound['domain'] = value

            return bound

        def check_bound(bound):
            min_reqs = ['name', 'type', 'domain']
            if set(bound.keys()) >= set(min_reqs):
                if hasattr(bound['domain'], 'rvs'):
                    bound['domain'] = distr_to_discrete(bound['domain'])
                    bound['type'] = 'discrete'
                    return bound
                elif isinstance(bound['domain'], Iterable):
                    return bound
                else:
                    raise TypeError('Domain is not iterable or Distribution')
            else:
                raise TypeError('Bound definition is not complete')

        def distr_to_discrete(distr):
            discrete_range = distr.rvs(n_samples).tolist()
            return discrete_range

        def check_str(bound):
            if isinstance(bound['domain'], list):
                if any(isinstance(s, (str, Iterable)) for s in bound['domain']):
                    str_configs[bound['name']] = bound['domain']
                    bound['domain'] = np.arange(len(bound['domain']))
                    return bound
            return bound

        bounds = []
        str_configs = {}
        if isinstance(candidate, Mapping):
            for param in candidate.keys():
                bounds += [check_str(param_to_bound(param, candidate[param]))]
            return bounds, str_configs
        elif isinstance(candidate, list):
            for _bound in candidate:
                bounds += [check_str(check_bound(_bound))]
            return bounds, str_configs
        else:
            raise TypeError('Invalid grid type')

    @staticmethod
    def _check_int(n):
        if isinstance(n, float):
            if n == int(n):
                return int(n)
        return n

    @disable
    def get_evaluations(self):
        pass

    @disable
    def run_optimization(self):
        pass

    @disable
    def plot_acquisition(self):
        pass

    @disable
    def plot_convergence(self):
        pass

    @disable
    def suggest_next_locations(self):
        pass

