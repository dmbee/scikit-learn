# Authors: David Burns <d.burns@utoronto.ca>
# License: BSD 3 clause

from collections import Counter
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state
from ..resample.base import ResamplerMixin

__all__ = ['RandomOverSampler']

class OverSamplerBase(ResamplerMixin):
    '''
    Base class for Resamplers that oversample

    Parameters
    ----------
    min_ratio : dict or float, range [0.0, 1.0] (default=1.0)
         Minimum ratio of samples in minority classes to the majority class(es).
         Can be specified as a float which will apply the min_ratio to all classes.
         Otherwise, min_ratio can be specified as a dict, where the values of min_ratio are applied
         individually to target classes based on the dict keys.

    accept_sparse : boolean (default=false)
        accept sparse matrix formats as input to resampler - this should be hard coded in derived
        classes depending if the resampler can handle sparse matrices

    validate : boolean (default=true)
        validate data passed to fit_resample

    '''
    def __init__(self, min_ratio = 1.0, accept_sparse=False, validate=True):
        super(OverSamplerBase, self).__init__(accept_sparse, validate)
        self.min_ratio = min_ratio

    def _get_oversampling(self, y):
        '''
        Returns the number of additional samples required for each class

        Parameters
        ----------
        y : array-like, shape [n_samples, ]
            the target vector

        Returns
        -------
        oversampling : dict
            the number of additional samples required for each class
        '''
        counts = Counter(y)

        if isinstance(self.min_ratio, dict):
            if not np.any([self.min_ratio[k] > 1.0 or self.min_ratio[k] < 0.0
                           for k in self.min_ratio]):
                raise ValueError("min_ratio parameter must be inside range [0.0, 1.0] for all classes")
        else:
            if self.min_ratio < 0.0 or self.min_ratio > 1.0:
                raise ValueError("min_ratio parameter must be inside range [0.0, 1.0]")
            self.min_ratio = {k: self.min_ratio for k in counts.keys()}

        N_max = counts.most_common()[0][1]

        oversampling = {k : int(self.min_ratio[k] * N_max - np.count_nonzero(y == k))
                for k in self.min_ratio.keys() & counts.keys()}

        return {k : oversampling[k] for k in oversampling if oversampling[k] > 0}


class RandomOverSampler(BaseEstimator, OverSamplerBase):
    '''
    Parameters
    ----------
    min_ratio : dict or float, range [0.0, 1.0] (default=1.0)
         Minimum ratio of samples in minority classes to the majority class(es).
         Can be specified as a float which will apply the min_ratio to all classes.
         Otherwise, min_ratio can be specified as a dict, where the values of min_ratio are applied
         individually to target classes based on the dict keys.

    shuffle : boolean (default=True)
        Whether or not to shuffle the data after resampling. If this is false, the samples
        are ordered by target class which is problematic for any batch optimizers used in
        model fitting.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    validate : boolean (default=true)
        validate data passed to fit_resample

    '''
    def __init__(self, min_ratio = 1.0, shuffle=True, random_state=None, validate=True):
        super(RandomOverSampler, self).__init__(min_ratio, accept_sparse=False, validate=validate)
        self.random_state = random_state
        self.shuffle = shuffle

    def _fit_resample(self, X, y, props=None):
        '''
        Resample the data

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

        rng = check_random_state(self.random_state)
        N_os = self._get_oversampling(y)
        samples = np.arange(len(y))
        samples = np.concatenate([samples] +
                                 [self._oversample(samples[y==k], N_os[k], rng) for k in N_os])

        if self.shuffle:
            samples = rng.shuffle(samples)

        return self._resample_from_indices(X, y, props, samples)


    def _oversample(self, samples, N, rng):
        ''' random oversampling to size N_min '''
        return rng.choice(samples, size=N, replace=True)








