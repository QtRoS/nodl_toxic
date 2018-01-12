import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split


def mean_log_loss(y_true, y_pred):
    """
    Mean value of per-label logloss (for multilabel classification)
    :param y_true: true classification (y_true[i, j] = 1 if i-th sample has j-th class, else 0)
    :type y_true: np.ndarray
    :param y_pred: predicted probabilities (y_true[i, j] ~= 1 if i-th sample has j-th class, else 0)
    :type y_pred: np.ndarray
    :return: loss value
    :rtype: float
    """
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    columns = y_pred.shape[1]
    errors = np.zeros([columns])
    for i in range(columns):
        errors[i] = log_loss(y_true[:, i], y_pred[:, i])
    return errors.mean()


def multilabel_label_combinations(y, n_splits):
    """
    Combination of labels for multilabel classification (for stratified sataset splitting)
    :param y: true classification (y_true[i, j] = 1 if i-th sample has j-th class, else 0)
    :type y: np.ndarray
    :param n_splits: split counts for validation (we'll need at least this count of combination samples)
    :type n_splits: int
    :return: possible label combinations. E.g. for two labels can be (if all combinations presented) \
        [[0 0]
         [0 1]
         [1 0]
         [1 1]]
    :rtype: np.ndarray
    """
    assert len(y.shape) == 2

    def _possible_label_combinations():
        columns = y.shape[1]
        combination_count = 2 ** columns
        combinations = np.zeros([combination_count, columns])
        idx = np.arange(combination_count, dtype=np.int)
        for i in range(columns):
            step = 2 ** i
            step_idx = ((idx // step) % 2) == 1
            combinations[step_idx, columns - i - 1] = 1.0
        return combinations

    def _analyzable_label_combinations():
        combinations = _possible_label_combinations()
        counts = np.fromiter(map(
            lambda i: np.all(y == combinations[i], axis=1).sum(),
            range(len(combinations))
        ), dtype=np.int)
        idx = counts >= n_splits
        return combinations[idx]

    return _analyzable_label_combinations()


def multilabel_cross_validation(classifier, X, y, n_splits=3, scoring=mean_log_loss, random_state=None):
    """
    Multilabel task crossvalidation
    :param classifier: classifier
    :type classifier: BaseEstimator
    :param X: dataset
    :param y: labels (y_true[i, j] = 1 if i-th sample has j-th class, else 0)
    :param n_splits: split count
    :type n_splits: int
    :param scoring: scorer function
    :type scoring: (np.ndarray, np.ndarray)->float
    :param random_state: random state for splitting
    :type random_state: int
    :return: scores
    :rtype: np.ndarray
    """
    def _convert_y(y):
        label_combinations = multilabel_label_combinations(y, n_splits)
        y_converted = np.zeros([len(y)])
        for i, combination in enumerate(label_combinations):
            y_converted[np.all(y == combination, axis=1)] = i
        return y_converted

    def _split(X, y_converted, n_splits, random_state):
        idx = np.arange(len(X), dtype=np.int)
        if n_splits == 1:
            train_idx, test_idx, _, _ = train_test_split(X, y_converted,
                                                         stratify=y_converted,
                                                         random_state=random_state)
            return [(train_idx, test_idx)]
        else:
            return StratifiedKFold(n_splits=n_splits, random_state=random_state).split(idx, y_converted)

    X = np.array(X)
    y = np.array(y)
    y_converted = _convert_y(y)
    base_params = classifier.get_params()
    if getattr(classifier, 'predict_proba', None) is None:
        predict = classifier.predict
    else:
        predict = classifier.predict_proba
    scores = []
    for train, test in _split(X, y_converted, n_splits, random_state):
        classifier.set_params(**base_params)
        classifier.fit(X[train], y[train])
        score = scoring(y[test], predict(X[test]))
        scores += [score]
    return np.array(scores)
