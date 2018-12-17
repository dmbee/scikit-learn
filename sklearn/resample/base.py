# Authors: David Burns <d.burns@utoronto.ca>
# License: BSD 3 clause

from abc import abstractmethod

from ..utils import check_array, check_consistent_length, check_X_y

class ResamplerMixin(object):
    '''
    Abstract base class for resamplers that includes code for checking input data and resampling
    using indices.

    Derived classes inheriting ``ResamplerMixin`` must implement ``_fit_resample``

    Parameters
    ----------
    accept_sparse : boolean
        accept sparse matrix formats as input to resampler
    validate : boolean
        validate data passes to fit_resample
    '''


    def __init__(self, accept_sparse=False, validate=True):
        self.shuffle = shuffle
        self.accept_sparse = accept_sparse
        self.validate = validate
        self._check_params()

    def _check_params(self):
        ''' validate class parameters '''
        if not type(self.shuffle) is bool:
            raise ValueError("shuffle parameter must be boolean type")
        if not type(self.validate) is bool:
            raise ValueError("validate parameter must be boolean type")

    def fit_resample(self, X, y, props=None, **fit_params):  # gets called by pipeline._fit()
        '''
        Resample X, y, and props

        Parameters
        ----------
        X : array-like, shape [n_samples, ]
            the data
        y : array-like, shape [n_samples, ]
            the target
        props : dictionary, each item array-like, shape [n_samples, ]
            sample properties (attributes/properties mapped to the data samples eg sample_weight)

        Returns
        -------
        X : array-like, shape [r_samples, ]
            resampled data
        y : array-like, shape [r_samples, ]
            resampled target
        props : dictionary, each item array-like, shape [e_samples, ]
            resampled sample properties

        '''
        if self.validate:
            X, y, props = self._check_data(X, y, props)
        return self._fit_resample(X, y, props, **fit_params)

    @abstractmethod
    def _fit_resample(self, X, y, props=None):
        '''
        Abstract method to be defined in each resampler which implements resampling logic

        Parameters
        ----------
        X : array-like, shape [n_samples, ]
            the data
        y : array-like, shape [n_samples, ]
            the target
        props : dictionary, each item array-like, shape [n_samples, ]
            sample properties (attributes/properties mapped to the data samples eg sample_weight)

        Returns
        -------
        X : array-like, shape [r_samples, ]
            resampled data
        y : array-like, shape [r_samples, ]
            resampled target
        props : dictionary, each item array-like, shape [e_samples, ]
            resampled sample properties

        '''
        return X, y, props

    def _check_data(self, X, y, props):
        ''' Checks and returns validated input data '''
        X, y = check_X_y(X, y, accept_sparse=self.accept_sparse, dtype=None)
        if props is not None:
            check_consistent_length(y, *list(props.values()))
            props = {k: check_array(props[k], accept_sparse=self.accept_sparse, dtype=None) for k in props}
        return X, y, props

    def _resample_from_indices(self, X, y, props, indices):
        ''' base method for resampling data from indices - can be called from inheriting classes '''
        if props is not None:
            return X[indices], y[indices], {k: props[k][indices] for k in props}
        else:
            return X[indices], y[indices], None
