from vegas_params import integral, expression
from vegas_params import Uniform, Fixed
import numpy as np

def assert_close(x, value, nsigma=1):
    assert np.abs(x.mean-value) < x.sdev*nsigma

def test_1d_constant():
    @integral
    @expression(x=Uniform([0,5]))
    def e(x):
        return np.ones(x.shape[0])

    assert_close(e(nitn=50), 5)

def test_Gaussian():
    @integral
    @expression(x=Uniform([-100,100]))
    def e(x):
        return np.exp(- x**2)

    assert_close(e(nitn=50), np.sqrt(np.pi))

