import pytest
import numpy as np
from collections import Counter

from .. import oversample

def all_oversamplers():
    return [(k, getattr(oversample, k)) for k in oversample.__all__
            if isinstance(getattr(oversample, k), oversample.OverSamplerBase)]

@pytest.mark.parametrize(
    ['name , Estimator'],
    all_oversamplers()
)
def test_oversamplers(name, Estimator):
    est = Estimator()
    check_non_sparse(Estimator)
    if est.accept_sparse:
        check_sparse(Estimator)

@pytest.mark.parametrize(
    [('min_ratio')],
    [(0.0),(1.0),(0.5),({0:1.0, 1:0.5, 2:1.0})]
)
def check_non_sparse(Estimator, min_ratio):

    est = Estimator(min_ratio)

    N = 30
    X = np.random.rand(N)
    y = np.concatenate([np.full(20, 0), np.full(7, 1), np.full(3, 2)])
    props = {'p': np.random.rand(N)}
    counts = dict(Counter(y))
    Nmax = max(counts.items())

    Xr, yr, propsr = est.fit_resample(X, y, props)
    countsr = dict(Counter(yr))

    if not isinstance(min_ratio, dict):
        assert np.all([countsr[k] >= int(Nmax * min_ratio[k]) for k in counts.keys() & min_ratio.keys()])
    else:
        assert np.all([countsr[k] >= int(Nmax * min_ratio) for k in counts.keys() & min_ratio.keys()])



def check_sparse(Estimator):
    est = Estimator()

