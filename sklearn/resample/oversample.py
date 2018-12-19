# Authors: David Burns <d.burns@utoronto.ca>
# License: BSD 3 clause

from collections import Counter
import numpy as np

from ..utils import check_random_state
from ..resample.base import ResamplerMixin

class RandomOverSampler(ResamplerMixin):
    '''
    Parameters
    ----------
    min_ratio : float, range [0.0, 1.0]
        Minimum ratio of samples in minority classes to the majority class(es).
        At the default of 1.0, all classes are resampled to have the same number of samples
        as the majority class.

    random_state : # todo

    accept_sparse : boolean
        accept sparse matrix formats as input to resampler
    validate : boolean
        validate data passes to fit_resample
    validate

    '''
    def __init__(self, min_ratio = 1.0, random_state=None, shuffle=True, accept_sparse=False, validate=True):
        super().__init__(accept_sparse, validate)
        self.min_ratio = min_ratio
        self.random_state = random_state
        self.shuffle = shuffle

    def _validate_params(self):
        ''' validate class parameters '''
        if self.min_ratio > 1.0 or self.min_ratio < 0.0:
            raise ValueError("min_ratio parameter must be None or inside range [0.0,1.0]")
        if not type(self.shuffle) is bool:
            raise ValueError("shuffle parameter must be boolean type")

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

        random_state = check_random_state(self.random_state)

        counts = Counter(y)
        N_min = counts.most_common()[0][1] * self.min_ratio
        counts = dict(counts)
        samples = np.arange(len(y))
        samples = np.concatenate([self._oversample(samples[y == yi], N_min)
                                  for yi in counts if counts[yi] < N_min])
        if self.shuffle:
            samples = np.random.shuffle(samples)

        return self._resample_from_indices(X, y, props, samples)


    def _oversample(self, samples, N_min):
        ''' random oversampling to size N_min '''
        Nos = min(N_min - len(samples), 1)
        return np.append(samples, np.random.choice(samples, size=Nos, replace=True))








