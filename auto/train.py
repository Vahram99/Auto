import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
from tune_sklearn import TuneSearchCV
from sklearn.model_selection import RepeatedKFold

import os


# This is a generic algorithm to find the best model
# Includes 3 types of models XGBoost, CatBoost and RandomForest
# includes hyperparameter tuning  using TuneSearchCV and GridSearchCV
# TuneSearchCV has features as search_optimization, early_stopping and gpu_usage
# includes also model combinations such as voting and stacking
# Cross-validation is used in all stages of model-evaluation
# Can be used for Classification and Regression tasks


class Train():
    def __init__(self, task, cv=5, cv_repeats=1, scoring='roc_auc', refit=True, n_jobs=-1, only_cat=False,
                 calibrate=False, ray=True, use_gpu=False, search_optimization='bayesian', early_stopping=True,
                 reduce_coef=10, grid_mode='light'):
        self.task = task
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        self.only_cat = only_cat
        self.calibrate = calibrate
        self.use_gpu = use_gpu
        self.ray = ray  # This param activates TuneSearchCV.If set to False, GridSearchCV is used
        self.search_optimization = search_optimization
        self.early_stopping = early_stopping
        self.reduce_coef = reduce_coef  # Reduces the initial grid by the coef. Search optimization allows to do that without seriously affecting the efficiency
        self.grid_mode = grid_mode
        self.cv_repeats = cv_repeats

    def models_and_grids(self, weights):
        if self.cv_repeats > 1:
            self.cv = RepeatedKFold(n_splits=self.cv, n_repeats=self.cv_repeats, random_state=42)
        if self.grid_mode == 'light':
            catboost_grid = {'max_depth': [6, 7, 8, 10], 'n_estimators': [50, 100, 300, 500, 1000]}
            xgboost_grid = {'max_depth': [6, 8, 10, 12, 15], 'n_estimators': [50, 100, 200, 300, 500],
                            'colsample_bytree': [0.3, 1], 'learning_rate': [0.01, 0.1]}
            rf_grid = {'max_depth': [6, 8, 10, 12, 15, 20], 'min_samples_split': [10, 20, 30, 40],
                       'min_samples_leaf': [5, 10], 'n_estimators': [100, 300]}


        elif self.grid_mode == 'hardcore':
            catboost_grid = {'max_depth': [6, 7, 8, 10],
                             'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000],
                             'learning_rate': [0.01, 0.05, 0.1]}
            xgboost_grid = {'max_depth': [6, 8, 10, 12, 15], 'n_estimators': [50, 100, 200, 300, 400, 500],
                            'colsample_bytree': [0.1, 0.3, 0.7, 1], 'learning_rate': [0.01, 0.05, 0.1],
                            'reg_alpha': [1, 3, 5, 10]}
            rf_grid = {'max_depth': [6, 8, 10, 12, 15, 20], 'min_samples_split': [10, 20, 30, 40],
                       'min_samples_leaf': [5, 7, 10], 'n_estimators': [50, 100, 300, 500]}

        if self.task == 'classification':
            catboost_model = CatBoostClassifier(random_state=116, cat_features=self.cat_features, class_weights=weights,
                                                thread_count=self.n_jobs, allow_writing_files=False)
            if np.unique(self.y_train).size > 2:
                xgboost_model = XGBClassifier(seed=116,
                                              n_jobs=self.n_jobs)
            else:
                xgboost_model = XGBClassifier(seed=116, scale_pos_weight=weights[1] / weights[0],
                                              n_jobs=self.n_jobs)
            rf_model = RandomForestClassifier(random_state=116, class_weight='balanced',
                                              n_jobs=self.n_jobs)
        elif self.task == 'regression':
            catboost_model = CatBoostRegressor(random_state=116, cat_features=self.cat_features,
                                               thread_count=self.n_jobs)
            xgboost_model = XGBRegressor(seed=116,
                                         n_jobs=self.n_jobs)
            rf_model = RandomForestRegressor(random_state=116, n_estimators=300,
                                             n_jobs=self.n_jobs)
        else:
            raise ValueError('Wrong task type')

        if self.ray:
            os.environ[
                "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # Fixes some errors related to the writing and reading of the temporary files
            os.environ[
                "RAY_DISABLE_MEMORY_MONITOR"] = "1"  # When more than 95% of Memory is used ray throws an error. This line disables that

        self.catboost_grid = catboost_grid
        self.xgboost_grid = xgboost_grid
        self.rf_grid = rf_grid
        self.catboost_model = catboost_model
        self.xgboost_model = xgboost_model
        self.rf_model = rf_model

    def gridsearch(self):
        def transform_to_TuneSearch(GridSearch):
            gridsearch_params = GridSearch.get_params()
            if self.use_gpu:
                gridsearch_params['n_jobs'] = 1
            param_grid = gridsearch_params['param_grid']

            combinations = 1
            for param in param_grid.keys():
                combinations *= len(param_grid[param])

            global_limit = min(combinations, 10)
            n_trials = max(global_limit, int(combinations / self.reduce_coef))
            n_trials = min(n_trials, 50)

            tune_search = TuneSearchCV(estimator=gridsearch_params['estimator'],
                                       param_distributions=gridsearch_params['param_grid'], use_gpu=self.use_gpu,
                                       search_optimization=self.search_optimization,
                                       early_stopping=self.early_stopping, n_jobs=gridsearch_params['n_jobs'],
                                       cv=self.cv, refit=self.refit, scoring=self.scoring,
                                       verbose=2, n_trials=n_trials, random_state=42)

            return tune_search

        if self.only_cat:
            grid_cat = GridSearchCV(estimator=self.catboost_model, param_grid=self.catboost_grid, scoring=self.scoring,
                                    refit=self.refit,
                                    cv=self.cv, n_jobs=self.n_jobs, verbose=10)

            if self.ray:
                grid_cat = transform_to_TuneSearch(grid_cat)
            grid_cat.fit(self.x_train, self.y_train)
            self.grid_cat = grid_cat
        else:
            grid_cat = GridSearchCV(estimator=self.catboost_model, param_grid=self.catboost_grid, scoring=self.scoring,
                                    refit=self.refit,
                                    cv=self.cv, n_jobs=self.n_jobs, verbose=10)

            grid_xgb = GridSearchCV(estimator=self.xgboost_model, param_grid=self.xgboost_grid, scoring=self.scoring,
                                    refit=self.refit, cv=self.cv,
                                    n_jobs=self.n_jobs, verbose=10)
            grid_rf = GridSearchCV(estimator=self.rf_model, param_grid=self.rf_grid, scoring=self.scoring,
                                   refit=self.refit,
                                   cv=self.cv, n_jobs=self.n_jobs,
                                   verbose=10)

            if self.ray:
                grid_cat = transform_to_TuneSearch(grid_cat)
                grid_xgb = transform_to_TuneSearch(grid_xgb)

            grid_cat.fit(self.x_train, self.y_train)
            grid_rf.fit(self.x_train, self.y_train)

            if np.unique(self.y_train).size > 2:
                sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
                grid_xgb.fit(self.x_train, self.y_train, sample_weight=sample_weight)
            else:
                grid_xgb.fit(self.x_train, self.y_train)

            self.grid_cat = grid_cat
            self.grid_xgb = grid_xgb
            self.grid_rf = grid_rf

    def voting(self):
        estimators = [('cat', self.grid_cat.best_estimator_), ('rf', self.grid_rf.best_estimator_),
                      ('xgb', self.grid_xgb.best_estimator_)]

        if self.task == 'classification':
            voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=self.n_jobs)
        else:
            voter = VotingRegressor(estimators=estimators, n_jobs=self.n_jobs)

        voter.fit(self.x_train, self.y_train)

        if type(self.scoring) is dict:
            voting_cvs = cross_validate(voter, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring,
                                        n_jobs=self.n_jobs)
        else:
            voting_cvs = cross_val_score(voter, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring,
                                         n_jobs=self.n_jobs)

        self.voter = voter
        self.voting_cvs = voting_cvs

    def stacking(self):
        estimators = [('cat', self.grid_cat.best_estimator_), ('rf', self.grid_rf.best_estimator_),
                      ('xgb', self.grid_xgb.best_estimator_)]

        if self.task == 'classification':
            stacker = StackingClassifier(estimators=estimators,
                                         final_estimator=LogisticRegression(class_weight='balanced'), cv=self.cv)
        else:
            stacker = StackingRegressor(estimators=estimators,
                                        final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                                        cv=self.cv
                                        )
        try:
            stacker.fit(self.x_train, self.y_train)
            if type(self.scoring) is dict:
                try:
                    stacking_cvs = cross_validate(stacker, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring,
                                                  n_jobs=self.n_jobs)
                except:
                    stacking_cvs = cross_validate(stacker, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring)
            else:
                try:
                    stacking_cvs = cross_val_score(stacker, self.x_train, self.y_train, cv=self.cv,
                                                   scoring=self.scoring, n_jobs=self.n_jobs)
                except:
                    stacking_cvs = cross_val_score(stacker, self.x_train, self.y_train, cv=self.cv,
                                                   scoring=self.scoring)
        except:
            stacker = self.grid_cat.best_estimator_
            stacking_cvs = self.grid_cat.best_score_

        self.stacker = stacker
        self.stacking_cvs = stacking_cvs

    def calibration(self):
        if self.x_train.shape[0] < 1000:
            method = 'sigmoid'
        else:
            method = 'isotonic'
        calibrator = Calibrate(self.best_model, self.x_train, self.y_train, method=method)
        cc = calibrator.get_cc()

        self.cc = cc

    def best_model(self):
        if self.only_cat:
            self.best_model = self.grid_cat.best_estimator_
            self.validation = self.grid_cat.best_score_
        else:
            models = [self.grid_cat.best_estimator_, self.grid_xgb.best_estimator_,
                      self.grid_rf.best_estimator_, self.voter, self.stacker]
            val_results = [self.grid_cat.best_score_, self.grid_xgb.best_score_, self.grid_rf.best_score_,
                           np.mean(self.voting_cvs), np.mean(self.stacking_cvs)]
            best_index = np.array(val_results).argmax()
            best_model = models[best_index]

            validation = {}
            for model_name, validation_result in zip(['cat', 'xgb', 'rf', 'voting', 'stacking'], val_results):
                validation[model_name] = validation_result

            self.best_model = best_model
            self.validation = validation

    def fit(self, x_train, y_train, cat_features=None):
        self.x_train = x_train
        self.y_train = y_train
        self.cat_features = cat_features

        if self.y_train.size <= 10000:
            self.early_stopping = False

        if self.task == 'classification':
            weights = class_weights(y_train)
        else:
            weights = None

        Train.models_and_grids(self, weights)
        if self.only_cat:
            Train.gridsearch(self)
        else:
            Train.gridsearch(self)
            Train.voting(self)
            Train.stacking(self)

        Train.best_model(self)

        self.grid_cat_results_ = self.grid_cat.cv_results_
        self.grid_rf_results_ = self.grid_rf.cv_results_
        self.grid_xgb_results_ = self.grid_xgb.cv_results_

        self.cat = self.grid_cat.best_estimator_
        self.xgb = self.grid_xgb.best_estimator_
        self.rf = self.grid_rf.best_estimator_

        if self.calibrate:
            Train.calibration(self)

        remove_temporary_attrs(self, leave=['best_model', 'validation', 'grid_cat_results_', 'grid_rf_results_',
                                            'grid_xgb_results_', 'voter', 'stacker', 'cat', 'xgb', 'rf'])

    @property
    def validation_results_(self):
        return self.validation

    @property
    def best_model_(self):
        return self.best_model

    @property
    def cc_(self):
        return self.cc


class Calibrate:
    def __init__(self, model, x_train, y_train, cv=5, n_jobs=-1, method='isotonic'):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.cv = cv
        self.n_jobs = n_jobs
        self.method = method

    def get_cc(self):
        base_estimator = clone(self.model)
        cc = CalibratedClassifierCV(base_estimator=base_estimator, cv=self.cv, n_jobs=self.n_jobs, method=self.method)
        cc.fit(self.x_train, self.y_train)
        self.cc = cc
        return cc

    def curves(self, x_test, y_test, n_bins=20):
        pred1 = self.model.predict_proba(x_test)[:, 1]
        pred2 = self.cc.predict_proba(x_test)[:, 1]
        prob_true1, prob_pred1 = calibration_curve(y_test,
                                                   pred1, n_bins=n_bins)
        prob_true2, prob_pred2 = calibration_curve(y_test,
                                                   pred2, n_bins=n_bins)

        plt.figure(figsize=(10, 5))
        plt.plot(prob_pred1, prob_true1, c='r')
        plt.plot(prob_pred2, prob_true2, c='g')
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))


def remove_temporary_attrs(obj, leave):
    all_attrs = list(obj.__dict__.keys())

    for attr in leave:
        if attr in all_attrs:
            all_attrs.remove(attr)

    for attr in all_attrs:
        delattr(obj, attr)


def class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    return class_weights
