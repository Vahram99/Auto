import os
import pandas as pd
from itertools import product
from auto.models import *
import pickle


parent_dir = "https://raw.githubusercontent.com/Vahram99/Auto/main/"

#Loading files
for x_y, train_test in product(['x', 'y'], ['train', 'test']):
    child_dir = f'{x_y}_{train_test}.csv'
    url = os.path.join(parent_dir, child_dir)
    globals()[f'{x_y}_{train_test}'] = pd.read_csv(url, on_bad_lines='skip')

cat_features = pd.read_csv(os.path.join(parent_dir, 'cat_features.csv'),
                           on_bad_lines='skip').values.flatten().tolist()

cat = CatBoost(task='cl', scoring='roc_auc', search_mode='bayesian',
               grid_mode='hardcore', search_verbosity=1, model_verbosity=0,
               n_jobs=-1, init_trials=50, cv_repeats=2,
               eps=0, max_time=11*3600)

cat.fit(x_train, y_train, cat_features=cat_features)

wrt_dir = '/home/loveslayer/Auto/results.pkl'

with open(wrt_dir, 'wb') as file:
    pickle.dump(cat, file)

