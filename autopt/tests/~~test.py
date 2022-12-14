import pickle

wrt_dir = '/home/loveslayer/Auto/results1.pkl'

with open(wrt_dir, 'rb') as f:
    results = pickle.load(f)

print(results.top_estimators(3, top_method='seq_furthest'))