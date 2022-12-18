from functools import partial

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

from ..methods.clustering import clustering
from ..methods.sequential import sequential
from ..utils import hasarg


METHODS = {'kmeans_exact': partial(clustering, model=KMeans, how='exact'),
           'kmeans_auto':  partial(clustering, model=KMeans, how='auto'),
           'kmodes_exact': partial(clustering, model=KModes, how='exact'),
           'kmodes_auto':  partial(clustering, model=KModes, how='auto'),
           'seq_best':     partial(sequential, init='best'),
           'seq_random':   partial(sequential, init='random'),
           'seq_furthest': partial(sequential, init='furthest')}


def _get_top_method(top_method, n_jobs=1, pre_dispatch='2*n_jobs'):
    if callable(top_method):
        return top_method

    if isinstance(top_method, str):
        if top_method in METHODS:
            method = METHODS[top_method]
            if hasarg(method, 'n_jobs'):
                method = partial(method, n_jobs=n_jobs,
                                 pre_dispatch=pre_dispatch)
            return method
        raise ValueError(f'Method {top_method} is not found')

    raise TypeError('top_method attribute must be str or callable')


def get_top_estimators(get_top, results_package, top_method=None,
                       candidate_span=None, n_jobs=1, pre_dispatch="2*n_jobs"):
    """Selects top n estimators from the search space

       Parameters
       ----------

       get_top: int
          Number of top estimators to select

       results_package: dict
          Dictionary of the search results
          Must contain at least
          - cv_results_ - ref:SearchBase.cv_results_
          - best_score_ - ref:SearchBase.best_score
          - base_estimator_ ref:SearchBase.base_estimator_

       top_method: str or callable, default=None
          Method for picking the top estimators
          ref: "auto.methods"

          top_method can be
          -  None          - simply selects n=get_top best performing estimators
          - 'kmeans_exact' - all estimators are clustered into exactly n=get_top clusters with KMeans
                             and best performing estimators are picked in each cluster
          - 'kmeans_auto'  - best number of clusters is computed automatically in the range [2,get_top]
                             KMeans clustering is used
          - 'kmodes_exact' - all estimators are clustered into exactly n=get_top clusters with KModes
                             and best performing estimators are picked in each cluster. Can be used only
                             for estimators that don't have method "predict_proba"
          - 'kmodes_auto'  - best number of clusters is computed automatically in the range [2,get_top]
                             KModes clustering is used. Can be used only for estimators that don't have
                             method "predict_proba"
          - 'seq_best'     - estimators are selected sequentially with the most different one picked at
                             each iteration. Best performing estimator is chosen as a starting point
          - 'seq_random'   - estimators are selected sequentially with the most different one picked at
                             each iteration. Starting point is chosen randomly
          - 'seq_furthest' - estimators are selected sequentially with the most different one picked at
                             each iteration. Chooses the furthest estimator as a starting point
          -  callable

       candidate_span: int, default=None
          Number of estimators to pick from
          None - defaults to value 5 * get_top

       n_jobs : int, default=1
          Number of jobs to run in parallel.
          "-1" means using all processors
          Active only if the method=top_method has the same argument

       pre_dispatch: int or str, default="2*n_jobs"
          Controls the number of jobs that get dispatched during parallel execution
          Active only if the method=top_method has the same argument

          """
    globals().update(results_package)

    params = np.array(cv_results_['params'])
    scores = cv_results_['mean_test_score']
    n_evaluations = len(params)

    if get_top > n_evaluations:
        raise ValueError('The number of top estimators can not be higher than '
                         'the total number of estimators')

    set_to_base = np.vectorize(lambda x: clone(base_estimator_).set_params(**x))

    if not top_method:
        top_indices = scores.flatten().argsort()[-get_top:]
        top_params = params[top_indices]
        top_estimators = set_to_base(top_params)

        return top_estimators

    if 'predictions' not in cv_results_:
        raise ValueError(f"Method '{top_method}' is not supported for this results_package")

    top_method = _get_top_method(top_method, n_jobs=n_jobs,
                                 pre_dispatch=pre_dispatch)

    predictions = cv_results_['predictions']

    if not candidate_span:
        candidate_span = 5 * get_top
    n_candidates = min(n_evaluations, candidate_span)

    indices = scores.flatten().argsort()[-n_candidates:].astype(int)
    candidate_preds = predictions[:, indices]
    candidate_scores = scores[indices]
    candidate_params = params[indices]

    top_indices = top_method(get_top=get_top, candidate_preds=candidate_preds,
                             candidate_scores=candidate_scores)
    top_params = candidate_params[top_indices]
    top_scores = scores[top_indices]
    top_estimators = set_to_base(top_params)

    out = dict(estimators=top_estimators,
               scores=top_scores)

    return out
