import numpy as np
from abc import abstractmethod

from ..utils import check_array, check_consistent_length, check_X_y

class ResamplerMixin(object):

    def __init__(self, accept_sparse=False, validate=True):
        self.accept_sparse = accept_sparse
        self.validate = validate
        self._check_params()

    def _check_params(self):
        if not type(self.accept_sparse) is bool:
            raise ValueError("accept_sparse parameter must be boolean type")
        if not type(self.validate) is bool:
            raise ValueError("validate parameter must be boolean type")

    def fit_resample(self, X, y, props=None, **fit_params):  # gets called by pipeline._fit()
        if self.validate:
            X, y, props = self._check_data(X, y, props)
        return self._fit_resample(X, y, props, **fit_params)

    @abstractmethod  # must be implemented in derived classes
    def _fit_resample(self, X, y, props=None, **fit_params):
        return X, y, props

    def _check_data(self, X, y, props):  # to be expanded upon
        X, y = check_X_y(X, y, accept_sparse=self.accept_sparse, dtype=None)
        if props is not None:
            check_consistent_length(y, *list(props.values()))
            props = {k: check_array(props[k], accept_sparse=self.accept_sparse, dtype=None) for k in props}
        return X, y, props

    def _resample_from_indices(self, X, y, props, indices):
        if props is not None:
            return X[indices], y[indices], {k: props[k][indices] for k in props}
        else:
            return X[indices], y[indices], None
