import pytest
import numpy as np

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
    ['Xsize', 'min_ratio'],
    [((20), 0.0),
     ((20), 1.0),
     ((20), 0.5),
     ((20,10), 0.0),
     ((20,10), 1.0),
     ((20,10), 0.5),
     ((50), {0:1.0, 1:0.5, 2:1.0})]
)
def check_non_sparse(Estimator, Xsize, min_ratio):
    est = Estimator(min_ratio)

    X = np.random.rand(*Xsize)
    y = np.random.randint(0,3,size=Xsize[0])
    props = {'p': np.random.randint(0,3,size=Xsize[0])}

    Xr, yr, propsr = est.fit_resample(X, y, props)



    if not isinstance(min_ratio, dict):
        assert np.all([np.count_nonzero(yr == yi) ])



def check_sparse(Estimator):
    est = Estimator()

