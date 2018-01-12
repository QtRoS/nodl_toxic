from  sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class MultilabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        """
        Multilabel classifier
        :param estimators: base estimators (one per label)
        :type estimators: list[BaseEstimator]
        """
        self.estimators = estimators

    def get_params(self, deep=True):
        return {
            'estimators': [
                estimator.get_params()
                for estimator in self.estimators
            ]
        }

    def set_params(self, **params):
        assert 'estimators' in params
        estimators_params = params['estimators']
        assert len(estimators_params) == len(self.estimators)
        for i, estimator in enumerate(self.estimators):
            estimator.set_params(**estimators_params[i])
        return self

    def fit(self, X, y):
        assert len(y.shape) == 2
        columns = y.shape[1]
        assert columns == len(self.estimators)
        for i, estimator in enumerate(self.estimators):
            estimator.fit(X, y[:, i])
        return self

    def predict(self, X):
        prediction = np.zeros([X.shape[0], len(self.estimators)])
        for i, estimator in enumerate(self.estimators):
            prediction[:, i] = estimator.predict(X)
        return prediction

    def predict_proba(self, X):
        prediction = np.zeros([X.shape[0], len(self.estimators)])
        for i, estimator in enumerate(self.estimators):
            if getattr(estimator, 'predict_proba', None) is None:
                predict = lambda X: estimator.predict(X)
            else:
                predict = lambda X: estimator.predict_proba(X)[:, 1]
            prediction[:, i] = predict(X)
        return prediction

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def fit_predict_proba(self, X, y):
        return self.fit(X, y).predict_proba(X)
