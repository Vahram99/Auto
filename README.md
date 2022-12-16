# AutOpt  

Automated hyperparameter optimization using BayesianSearchCV. Apart from the main features, AutOpt also provides larger functionality for cross-validation and several methods for selecting top n estimators from the search space. It supports regression and binary classification. 

## Models

There are predefined search spaces for each of the below estimators in 3 grid-modes -> **Light**, **Medium**, **Hardcore**

- SVM
- MLP
- RandomForest
- ExtraTrees
- HistGradienBoosting
- LGBM
- XGBoost
- CatBoost

The lighter the grid-mode the less hyperparameters are included in the search space, and their individual domains are also comparably narrower. The lighter the grid-mode the less iterations it needs to come to a reasonable result, but on larger numbers of iterations the wider search-spaces will generally perform better.

## Search

AutOpt supports hyperparameter search with ```BayesianSearchCV``` and ```RandomizedSearchCV```. The former one is developed on the base of ```GPyOpt.methods.BayesianOptimization``` and the design is inspired by sklearn's ```BaseSearchCV```. It preserves the main features of BaseSearchCV, while adding a few new ones, such as collecting cross-validation predictions (which is necessary for selecting top n estimators), saving intermediate results, or imposing time constraints. 

## Methods

- clustering
- sequential

AutOpt provides a tool for selecting the top n estimators based on the search results. The frst method ```clustering``` does this by separating the entire set of candidates into n clusters (based on their cross-validation predictions) and then picking the best performing estimators from those clusters. The ```sequential``` method does not prioratize the performance, instead it tries to find estimators most different from each other. This tool has many use cases in ensemble learning.

## Usage

Instantiating a hyperparameter optimization is as simple as importing the appropriate estimator from ```autopt.models``` and defining the task and the grid-mode.

```python
from autopt.models import CatBoost
from dataset import x, y  # Data

optimization = CatBoost(task='classification', grid_mode='medium')
optimization.fit(x, y)
```



The code is written in a fashion that allows much room for customization. For example, in order to create a new model in the likeness of the ones in ```autopt.models```, one would have to subclass the ```autopt.core.SearchBase``` and override two abstract methods.

```python
from autopt.core import SearchBase
from mymodule import MyClassifier, MyRegressor
from autopt.models.utils import _get_grid, _get_base

class MyModel(SearchBase):
    @staticmethod
    def _grid(grid_mode, shape=None):

    # At least one grid-mode is required
    grids = dict(
                light=dict("search space for 'light' mode"),
                medium=dict("search space for 'medium' mode"),
                hardcore=dict("search space for 'hardcore' mode")
                )

    @staticmethod
    def _estimator_base(task, n_jobs, verbosity):
        classifier = MyClassifier
        regressor = MyRegressor

        return _get_base(task, n_jobs, verbosity, classifier, regressor)
```

Not to tinker with the private functions and subclassing there is a constructor ```make_searcher``` that can do the job for you.

```python
from autopt.models import make_searcher
from myestimator import MyClassifier, MyRegressor
from dataset import x, y  # Data

classifier = MyClassifier
regressor  = MyRegressor

# At least one grid-mode is required
grids = dict(
             light=dict("search space for 'light' mode"),
             medium=dict("search space for 'medium' mode"),
             hardcore=dict("search space for 'hardcore' mode")
             )

MyModel = make_searcher(grids, classifier, regressor)
optimization = MyModel(task='regression', grid_mode='hardcore')
optimization.fit(x, y)
```

## Vision

Ultimately AutOpt is designed to simplify the hyperparameter tuning process, allowing to write as little code as possible, while preserving the ability to tweak that process at each step if needed.

Future features I would like to introduce

- Multiclass and multioutput classification
- New search algorithms such as HyperOpt
- Hyperparameter tuning for Neural Networks
- GPU support
