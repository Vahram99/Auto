import os
import pandas as pd
from itertools import product
from .models import *

parent_dir = "https://github.com/Vahram99/Auto/blob/main/"

#Loading files
for x_y, train_test in product(['x', 'y'], ['train', 'test']):
    child_dir = f'{x_y}_{train_test}.csv'
    url = os.path.join(parent_dir, child_dir)
    globals()[child_dir] = pd.read_csv(url, on_bad_lines='skip')

cat_features = pd.read_csv(os.path.join(parent_dir, 'cat_features.csv'))

cat = CatBoost(task='cl', scoring='roc_auc', search_mode='bayesian',
               grid_mode='hardcore', search_verbosity=1, model_verbosity=0,
               n_jobs=-1, init_trials=30, cv_repeats=2, max_time=8*3600, n_iter=1000, eps=0)

cat.fit(x_train, y_train, cat_features=cat_features)
