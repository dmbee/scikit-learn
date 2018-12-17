# Authors: David Burns <d.burns@utoronto.ca>
# License: BSD 3 clause

from ..utils import check_random_state
from ..resample.base import ResamplerMixin

class RandomOverSampler(ResamplerMixin):
    '''

    '''
    def __init__(self, shuffle=True, random_state=None, accept_sparse=False, validate=True):
        super().__init__(accept_sparse, validate)
        self.random_state = random_state
        self.shuffle = shuffle

    def _fit_resample(self, X, y, props=None):

        random_state = check_random_state(self.random_state)


