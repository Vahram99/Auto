import os
import pickle
import pandas as pd
from itertools import product
from autopt.models.trees import CatBoost


parent_dir = "https://raw.githubusercontent.com/Vahram99/Auto/main/"

# Loading files
for x_y, train_test in product(['x', 'y'], ['train', 'test']):
    child_dir = f'{x_y}_{train_test}.csv'
    url = os.path.join(parent_dir, child_dir)
    globals()[f'{x_y}_{train_test}'] = pd.read_csv(url, on_bad_lines='skip')

cat_features = pd.read_csv(os.path.join(parent_dir, 'cat_features.csv'),
                           on_bad_lines='skip').values.flatten()

wrt_dir = '/home/loveslayer/Auto/results1.pkl'


cat = CatBoost(task='cl', scoring='roc_auc', search_mode='random',
           grid_mode='light', search_verbosity=2, model_verbosity=0,
           n_jobs=-1, cv_repeats=1, n_iter=10, top_method='kmeans_auto', get_top=3)

cat.fit(x_train.values, y_train)

cat.save(path_to_file=wrt_dir, package=False)

import pickle
wrt_dir = '/home/loveslayer/Auto/results1.pkl'
with open(wrt_dir, 'rb') as f:
    results = pickle.load(f)

print(results.cv_results_['predictions'].shape)

