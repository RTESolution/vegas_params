from vegas_params import integral, expression
from vegas_params import Uniform, Fixed
import numpy as np

def assert_close(x, value, nsigma=1):
    assert np.abs(x.mean-value)<x.sdev*nsigma

def test_1d_constant():
    @integral
    @expression(x=Uniform([0,5]))
    def e(x):
        return np.ones(x.shape[0])

    assert_close(e(),5)


