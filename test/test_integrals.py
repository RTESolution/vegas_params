import numpy as np
from vegas_params import expression, Uniform, Vector, Direction, Scalar, vector
from vegas_params import integral

def assert_integral_is_close(e, value, precision=0.1):
    x = integral(e)(nitn=30, neval=10000)
    assert np.abs(x.mean - value) < value*precision
    
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

def test_Spherical_integral_simple():
    def density(r):
        return np.ones(r.shape[0])
    #test spherical integral with factor
    Rsphere=10
    @expression(R=Scalar(Uniform([0,Rsphere])), direction=Direction())
    def density(R, direction):
        r = R*direction
        return R**2

    assert_integral_is_close(density, 4/3*np.pi*Rsphere**3)

def test_Spherical_integral():
    #test spherical integral with factor
    @expression
    class Spherical:
        R:Scalar = Scalar(Uniform([0,1]))
        s:Direction = Direction()
        def __call__(self,R,s):
            self.factor = R**2
            return R*s

    Rsphere=10
    
    @expression(r=Spherical(R=Scalar(Uniform([0,Rsphere]))))
    def density(r:vector):
        return np.ones(r.shape[0])
        
    assert_integral_is_close(density, 4/3*np.pi*Rsphere**3)
