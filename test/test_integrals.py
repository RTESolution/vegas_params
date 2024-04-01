import numpy as np
from vegas_params import expression, Uniform, Vector, Direction, Scalar, vector
from vegas_params import integral

def assert_integral_is_close(e, value, nsigmas=1):
    x = integral(e)(nitn=10, neval=20000)
    assert (x.mean - value) < nsigmas*x.sdev
    
def test_1d_constant_integral():
    #test linear integral
    @expression(x=Uniform([0,5]))
    def constant(x):
        return np.ones(x.shape[0])

    assert_integral_is_close(constant,5)

def test_2d_constant_integral():
    @expression(x=Uniform([0,5]), y=Uniform([0,5]))
    def constant(x,y):
        return np.ones(x.shape[0])

    assert_integral_is_close(constant,25)

def test_Gaussian_integral():
    @expression(x=Uniform([-100,100]))
    def gaussian(x):
        return np.exp(-x**2)

    assert_integral_is_close(gaussian, np.sqrt(np.pi))

def test_Gaussian_integral():
    #test linear integral with factor
    
    @expression(x=Uniform([-100,100]))
    def gaussian(x):
        return np.exp(-x**2)

    assert_integral_is_close(gaussian, np.sqrt(np.pi))

def test_Spherical_integral():
    #test spherical integral with factor
    @expression
    class Spherical:
        R:Scalar = Scalar(Uniform([0,1]))
        s:Direction = Direction()
        def make(self,R,s):
            self.factor = R**2
            return R*s

    R=10
    
    @expression(r=Spherical(R=Uniform([0,R])))
    def density(r:vector):
        return np.ones(r.shape[0])
        
    assert_integral_is_close(density, 4/3*np.pi*R**3)