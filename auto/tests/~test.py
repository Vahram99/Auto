import pickle

from auto.base import get_top_estimators
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
import os
import pandas as pd
import numpy as np
from itertools import product

wrt_dir = '/home/loveslayer/Auto/results.pkl'

with open(wrt_dir, 'rb') as file:
    a = pickle.load(file)

a['base_estimator_'] = a['bese_estimator']

top_estimators = get_top_estimators(10, a, top_method='kmeans_auto',
                                    candidate_span=50, n_jobs=-1)

parent_dir = "https://raw.githubusercontent.com/Vahram99/Auto/main/"

#Loading files
for x_y, train_test in product(['x', 'y'], ['train', 'test']):
    child_dir = f'{x_y}_{train_test}.csv'
    url = os.path.join(parent_dir, child_dir)
    globals()[f'{x_y}_{train_test}'] = pd.read_csv(url, on_bad_lines='skip')

cat_features = pd.read_csv(os.path.join(parent_dir, 'cat_features.csv'),
                           on_bad_lines='skip').values.flatten().tolist()

l = len(top_estimators)
for i in range(l):
    top_estimators[i].set_params(**{'cat_features': cat_features})
    top_estimators[i] = (chr(i), top_estimators[i])

print(top_estimators)

"""
v = VotingClassifier(top_estimators.tolist(), voting='soft', n_jobs=-1)

s = cross_validate(v, x_train, y_train,
                   n_jobs=-1, scoring='roc_auc')

print(s)
print(s['test_score'].mean())"""