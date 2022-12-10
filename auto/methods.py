import types
import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

__all__ = ['_clustering']

CLUSTERING_METRICS = {'silhouette_score': (silhouette_score, 1),
                      'calinski_harabasz_score': (calinski_harabasz_score, 1),
                      'davies_bouldin_score': (davies_bouldin_score, -1)}


def _add_to_metrics(maximize):
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
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(arr.reshape(-1, 1))
        return scaled_arr.flatten()

    @staticmethod
    def _combine(raw):
        out = raw.pop(0)
        [[out[k].extend(scores[k]) for scores in raw] for k in out]

        return out


def _get_clustering_metric(metric):
    if isinstance(metric, tuple):
        if not callable(metric[0]):
            raise TypeError('Provided metric function is not a callable')
        if not isinstance(metric[1], int):
            raise TypeError(f"'maximize' must be integer, not {type(metric)}")
        if metric[1] not in (1, -1):
            raise ValueError("only -1 and 1 values expected for 'maximize'")
        return metric[0], metric[1]
    if isinstance(metric, str):
        metric, maximize = CLUSTERING_METRICS[metric]
        if not isinstance(metric, types.FunctionType):
            metric = metric()
            return metric, maximize
        return metric, maximize


def _clustering(model, get_top, candidate_preds, candidate_scores,
                how='exact', metric='combined_score', n_init=100,
                n_jobs=1, pre_dispatch="2*n_jobs"):
    def _clusterize(model, n_clusters, data, n_init):
        clustering = model(n_clusters=n_clusters, n_init=n_init,
                           random_state=np.random.randint(0, 1e+5))
        clustering.fit(data)
        return clustering.labels_

    def _evaluate(metric, x, labels):
        score = metric(x, labels)
        return score

    data = pd.DataFrame(candidate_preds.T)
    if how == 'exact':
        labels = _clusterize(model, get_top, data, n_init)
    elif how == 'auto':
        metric, maximize = _get_clustering_metric(metric)

        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        candidate_labels = parallel(
            delayed(_clusterize)(
                model,
                n_clusters,
                data,
                n_init
            )
            for n_clusters in range(2, get_top + 1)
        )

        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        scores = parallel(
            delayed(_evaluate)(
                metric,
                data,
                labels
            )
            for labels in candidate_labels
        )

        if hasattr(metric, 'get_scores'):
            scores = metric.get_scores(scores)
        scores = np.array(scores)
        best_index = scores.argsort()[::maximize][-1]
        labels = candidate_labels[best_index]
    else:
        raise ValueError('Invalid value for "how"')

    clusters = pd.DataFrame({'labels': labels, 'scores': candidate_scores})
    groups = clusters.groupby('labels')['scores']
    indices = groups.idxmax().values

    return indices.astype(int)
