from sklearn.base import BaseEstimator, TransformerMixin


class TransformPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, steps):
        self.steps = steps

    def get_params(self, deep=True):
        return {
            'steps': {
                name: step.get_params(deep)
                for name, step in self.steps
            }
        }

    def set_params(self, **params):
        assert 'steps' in params
        names = [name for name, _ in self.steps]
        assert set(names) == set(params['steps'].keys())
        for name, step_params in params['steps'].items():
            self.steps[names.index(name)][1].set_params(**step_params)
        return self

    def fit(self, X, y=None):
        transformed = X
        for _, step in self.steps:
            transformed = step.fit_transform(transformed, y)
        return self

    def transform(self, X, y=None):
        transformed = X
        for _, step in self.steps:
            transformed = step.transform(transformed)
        return transformed
