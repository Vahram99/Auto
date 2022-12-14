import numpy as np
from scipy.spatial.distance import pdist, squareform


def sequential(get_top, candidate_preds, candidate_scores,
               init='best', metric=None):
    """
    Method for selecting top n estimators from the search_space

    Selects top estimators sequentially using their predictions as the data,
    and at each iteration picks the estimator most different from the
    already selected ones

    Parameters
    ----------

    get_top: int
      Number of estimators to choose

    candidate_preds: list or numpy.ndarray
      Cross-validation predictions of candidate estimators

    candidate_scores: list or numpy.ndarray
      Cross-validation scores of candidate estimators

    init: str, default='best'
      Initialization strategy

      Can take values
      - "best"     - chooses the best performing estimator as a starting point
      - "random"   - chooses the starting point randomly
      - "furthest" - chooses the furthest estimator as a starting point

    metric: str, default=None
      Metric for computing the pairwise distances
      ref: scipy.spatial.distance.pdist

    """
    if not metric:
        # Whether candidate_preds are predictions or probas
        if (candidate_preds.astype(int) == candidate_preds).all():
            metric = 'hamming'
        else:
            metric = 'euclidean'

    def _initialize():
        # Set the starting point
        if init == 'best':
            best_idx = candidate_scores.argmax()
            return best_idx
        elif init == 'random':
            return np.random.randint(0, len(candidate_scores))
        elif init == 'furthest':
            return np.sum(square_pdist, axis=0).argmax()

    def _remove_from_candidates(idx):
        # Fills already selected columns with 0
        # This prevents choosing the same column over and over again
        square_pdist[:, idx].fill(0)

    def _next():
        _remove_from_candidates(selected[-1])

        # Distance between any point and set of selected points is the
        # sum of their pairwise distances
        distances = np.sum(square_pdist[selected, :], axis=0)
        out = distances.argmax()  # Selects the furthest point

        return out

    data = candidate_preds.T
    pairwise_distances = pdist(data, metric)

    # Converts a vector-form distance vector to a square-form distance matrix
    square_pdist = squareform(pairwise_distances)

    first = _initialize()
    selected = [first]

    while len(selected) < get_top:
        next_selected = _next()
        selected.append(next_selected)

    selected = np.array(selected).astype(int)

    return selected
