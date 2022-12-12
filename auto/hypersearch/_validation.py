import numbers
import time
from traceback import format_exc

import numpy as np
from joblib import Parallel, logger

from sklearn.base import is_classifier, clone
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.fixes import delayed
from sklearn.utils import indexable
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring

from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts, _score, _insert_error_scores
from sklearn.model_selection._validation import _warn_about_fit_failures, _normalize_score_results


def _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters,
                   fit_params, return_train_score=False, return_parameters=False,
                   return_n_test_samples=False, return_times=False, return_estimator=False,
                   return_predictions=False, split_progress=None, candidate_progress=None,
                   error_score=np.nan):

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer, error_score)
        if return_predictions:
            method = 'predict_proba' if hasattr(estimator,'predict_proba') else 'predict'
            func = getattr(estimator, method)
            predictions = func(X_test)
            if predictions.ndim > 1:
                predictions = predictions[:, 1]
        score_time = time.time() - start_time - fit_time

    result_msg = ""
    if verbose > 3:
        progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        total_time = score_time + fit_time
        end_msg = f"    [CV{progress_msg}] END "
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (60 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    if return_predictions:
        result['predictions'] = predictions
    return result


def cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
                   n_jobs=None, verbose=0, fit_params=None, pre_dispatch="2*n_jobs",
                   return_train_score=False, return_estimator=False, return_predictions=False,
                   error_score=np.nan):

    """
    Same function as sklearn.model_selection.cross_validate but with more functionality

    Evaluate metric(s) by cross-validation and also record fit/score times.

        Read more in the :ref:`User Guide <multimetric_cross_validation>`.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            The target variable to try to predict in the case of
            supervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`GroupKFold`).

        scoring : str, callable, list, tuple, or dict, default=None
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:

            - a single string (see :ref:`scoring_parameter`);
            - a callable (see :ref:`scoring`) that returns a single value.

            If `scoring` represents multiple scores, one can use:

            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric
              names and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.


        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For int/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`.Fold` is used. These splitters are instantiated
            with `shuffle=False` so the splits will be the same across calls.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.


        n_jobs : int, default=None
            Number of jobs to run in parallel. Training the estimator and computing
            the score are parallelized over the cross-validation splits.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : int, default=0
            The verbosity level.

        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        pre_dispatch : int or str, default='2*n_jobs'
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs

                - An int, giving the exact number of total jobs that are
                  spawned

                - A str, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        return_train_score : bool, default=False
            Whether to include train scores.
            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be computationally
            expensive and is not strictly required to select the parameters that
            yield the best generalization performance.

        return_estimator : bool, default=False
            Whether to return the estimators fitted on each split.

        return_predictions : bool, default=False
            Whether to return the predictions on each split.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised.
            If a numeric value is given, FitFailedWarning is raised.


        Returns
        -------
        scores : dict of float arrays of shape (n_splits,)
            Array of scores of the estimator for each run of the cross validation.

            A dict of arrays containing the score/time arrays for each scorer is
            returned. The possible keys for this ``dict`` are:

                ``test_score``
                    The score array for test scores on each cv split.
                    Suffix ``_score`` in ``test_score`` changes to a specific
                    metric like ``test_r2`` or ``test_auc`` if there are
                    multiple scoring metrics in the scoring parameter.
                ``train_score``
                    The score array for train scores on each cv split.
                    Suffix ``_score`` in ``train_score`` changes to a specific
                    metric like ``train_r2`` or ``train_auc`` if there are
                    multiple scoring metrics in the scoring parameter.
                    This is available only if ``return_train_score`` parameter
                    is ``True``.
                ``fit_time``
                    The time for fitting the estimator on the train
                    set for each cv split.
                ``score_time``
                    The time for scoring the estimator on the test set for each
                    cv split. (Note time for scoring on the train set is not
                    included even if ``return_train_score`` is set to ``True``
                ``estimator``
                    The estimator objects for each cv split.
                    This is available only if ``return_estimator`` parameter
                    is set to ``True``."""

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers,
            train_test[0],
            train_test[1],
            verbose,
            None,
            fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            return_predictions=return_predictions,
            split_progress=(n_split, cv.get_n_splits()),
            error_score=error_score,
        )
        for n_split, train_test in enumerate(cv.split(X, y, groups))
    )

    _warn_about_fit_failures(results, error_score)

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]
    if return_predictions:
        predictions = np.concatenate(results['predictions'], axis=0)
        ret["predictions"] = predictions

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    return ret