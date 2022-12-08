import numpy as np
import pandas as pd

__all__ = ['_clustering']


def _clustering(model, get_top, candidate_preds, candidate_scores, n_init=100):
    candidate_preds = pd.DataFrame(candidate_preds.T)

    km = model(n_clusters=get_top, random_state=np.random.randint(0, 1e+5),
               n_init=n_init)
    km.fit(candidate_preds)

    clusters = pd.DataFrame({'labels': km.labels_, 'scores': candidate_scores})
    groups = clusters.groupby('labels')['scores']
    indices = groups.idxmax().values



    return indices.astype(int)



