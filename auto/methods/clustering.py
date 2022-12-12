import types
from joblib import Parallel

import numpy as np
import pandas as pd

from sklearn.utils.fixes import delayed
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from ..utils import hasarg

CLUSTERING_METRICS = {'silhouette_score': (silhouette_score, 1),
                      'calinski_harabasz_score': (calinski_harabasz_score, 1),
                      'davies_bouldin_score': (davies_bouldin_score, -1)}


def _add_to_metrics(maximize):
    """
    Decorator for adding user_defined metrics
    to CLUSTERING_METRICS
    """
    def decorate(metric):
        if not callable(metric):
            raise TypeError('Provided metric must be a callable')
        name = metric.__name__
        metric_tuple = (metric, maximize)
        CLUSTERING_METRICS[name] = metric_tuple

        return metric

    return decorate


@_add_to_metrics(maximize=1)
class combined_score:
    """
    Metric for clustering evaluation that
    combines several metrics from CLUSTERING_METRICS
    """
    def __init__(self):
        pass

    def __call__(self, x, labels):
        return self._metric(x, labels)

    def get_scores(self, raw):
        combined_scores = self._combine(raw)
        scaled_scores = {k: CLUSTERING_METRICS[k][1] * self._scale(v) for k, v in combined_scores.items()}

        out = [sum(v) for v in zip(*scaled_scores.values())]
        return out

    @staticmethod
    def _metric(x, labels):
        score = {}
        for key in CLUSTERING_METRICS.keys():
            if key == combined_score.__name__:
                continue
            metric, _ = CLUSTERING_METRICS[key]
            score[key] = [metric(x, labels)]

        return score

    @staticmethod
    def _scale(arr):
        # Brings all metrics into the same scale
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(arr.reshape(-1, 1))
        return scaled_arr.flatten()

    @staticmethod
    def _combine(raw):
        # Merges the results of the parallel execution
        out = raw.pop(0)
        [[out[k].extend(scores[k]) for scores in raw] for k in out]

        return out


def _get_clustering_metric(metric):
    # Checks the syntax of the metric
    if isinstance(metric, tuple):
        if not callable(metric[0]):
            raise TypeError('Provided metric function is not a callable')
        if not isinstance(metric[1], int):
            raise TypeError(f"'maximize' must be integer, not {type(metric)}")
        if metric[1] not in (1, -1):
            raise ValueError("only -1 and 1 values expected for 'maximize'")
        return metric[0], metric[1]

    # Invokes the metric from CLUSTERING_METRICS
    if isinstance(metric, str):
        metric, maximize = CLUSTERING_METRICS[metric]
        if not isinstance(metric, types.FunctionType):
            metric = metric()  # if object the metric must be callable
            return metric, maximize
        return metric, maximize


def clustering(model, get_top, candidate_preds, candidate_scores,
               how='exact', metric='combined_score', labels_attr='labels_',
               n_jobs=1, pre_dispatch="2*n_jobs", **clustering_params):

    """Method for selecting top n estimators from the search_space

       Separates the set of candidate estimators into several clusters,
       using their predictions as the clustering data, and picks the top
       estimators from those clusters

       Parameters
       ----------

       model: estimator
          Clustering algorithm with parameter n_clusters

       get_top: int
          Number of estimators to choose

       candidate_preds: list or numpy.ndarray
          Cross-validation predictions of candidate estimators

       candidate_scores: list or numpy.ndarray
          Cross-validation scores of candidate estimators

       how: str, default='exact'
          How to select the number of top estimators
          - "exact" - n=get_top clusters are formed and exactly n estimators are selected
          - "auto"  - best number of clusters is computed automatically in range [2, get_top]

       metric: str or tuple, default='combined_score'
          Metric to evaluate the clustering
          Can be
          - 'silhouette_score', ref: "sklearn.metrics"
          - 'calinski_harabadz_score', ref: "sklearn.metrics"
          - 'davies_bouldin_score', ref: "sklearn.metrics"
          - 'combined_score' - combination of all the metrics in CLUSTERING_METRICS
          - callable - tuple in form (callable, 1 if maximize -1 if minimize)
          Active only if how="auto"

       labels_attr: str, default='labels_'
          Name of the attribute in the clustering model that returns
          the appropriate labels

       n_jobs : int, default=1
          Number of jobs to run in parallel.
          "-1" means using all processors

       pre_dispatch: int or str, default="2*n_jobs"
          Controls the number of jobs that get dispatched during parallel execution

       **clustering_params
          Params to pass to the model
       """
    def _set_param(param, value):
        if not any(arg in clustering_params for arg in default_params[param]):
            for arg in default_params[param]:
                if hasarg(model, arg):
                    clustering_params[arg] = value

    def _clusterize(n_clusters):
        # Performs the clustering
        _set_param('n_clusters', n_clusters)
        clustering_model = model(**clustering_params)
        clustering_model.fit(data)
        return eval(f"clustering_model.{labels_attr}")

    def _evaluate(labels):
        # Evaluates the clustering
        score = metric(data, labels)
        return score

    default_params = {'n_init': ('n_init', 'max_init'),
                      'n_clusters': ('n_clusters', 'n_components')}
    _set_param('n_init', 100)

    data = pd.DataFrame(candidate_preds.T)
    if how == 'exact':
        labels = _clusterize(n_clusters=get_top)
    elif how == 'auto':
        metric, maximize = _get_clustering_metric(metric)

        # Parallel training of clustering models
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        candidate_labels = parallel(
            delayed(_clusterize)(
                n_clusters=n_clusters,
            )
            for n_clusters in range(2, get_top + 1)
        )

        # Parallel evaluation of clustering models
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        scores = parallel(
            delayed(_evaluate)(
                labels=labels
            )
            for labels in candidate_labels
        )

        # User-defined metrics must have get_scores attribute (if object)
        # or return the score immediately (if function)
        if hasattr(metric, 'get_scores'):
            scores = metric.get_scores(scores)
        scores = np.array(scores)
        best_index = scores.argsort()[::maximize][-1]
        labels = candidate_labels[best_index]
    else:
        raise ValueError('Invalid value for "how"')

    clusters = pd.DataFrame({'labels': labels, 'scores': candidate_scores})
    groups = clusters.groupby('labels')['scores']
    indices = groups.idxmax().values  # Gets best performing estimator in each of the clusters

    return indices.astype(int)

