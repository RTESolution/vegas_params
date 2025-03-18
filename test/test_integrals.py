import numpy as np
import vegas_params as vp

def assert_integral_is_close(e, value, precision=0.1):
    x = vp.integral(e)(nitn=30, neval=10000)
    assert np.abs(x.mean - value) < value*precision
    
def test_1d_constant_integral():
    #test linear integral
    @vp.expression(x=vp.Uniform([0,5]))
    def constant(x):
        return np.ones(x.shape[0])

    assert_integral_is_close(constant,5)

def test_2d_constant_integral():
    @vp.expression(x=vp.Uniform([0,5]), y=vp.Uniform([0,5]))
    def constant(x,y):
        return np.ones(x.shape[0])

    assert_integral_is_close(constant,25)

def test_Gaussian_integral():
    #test linear integral with factor
    
    @vp.expression(x=vp.Uniform([-100,100]))
    def gaussian(x):
        return np.exp(-x**2)

    assert_integral_is_close(gaussian, np.sqrt(np.pi))

def test_Spherical_integral_simple():
    def density(r):
        return np.ones(r.shape[0])
    #test spherical integral with factor
    Rsphere=10
    @vp.expression(R=vp.Scalar(vp.Uniform([0,Rsphere])), direction=vp.Direction())
    def density(R, direction):
        r = R*direction
        return R**2

    assert_integral_is_close(density, 4/3*np.pi*Rsphere**3)

def test_Spherical_integral():
    #test spherical integral with factor
    @vp.expression
    class Spherical:
        R:vp.Scalar = vp.Scalar(vp.Uniform([0,1]))
        s:vp.Direction = vp.Direction()
        def __call__(self,R,s):
            self.factor = R**2
            return R*s

    Rsphere=10
    
    @vp.expression(r=Spherical(R=vp.Scalar(vp.Uniform([0,Rsphere]))))
    def density(r:vp.Vector):
        return np.ones(r.shape[0])
        
    assert_integral_is_close(density, 4/3*np.pi*Rsphere**3)

def test_Spherical_integral_normalized():
    #test spherical integral with factor
    @vp.expression
    class Spherical:
        R:vp.Scalar = vp.Scalar(vp.Uniform([0,1]))
        s:vp.Direction = vp.Direction()
        def __call__(self,R,s):
            self.factor = R**2
            return R*s

    Rsphere=10

    @vp.utils.normalize_integral(123)
    @vp.expression(r=Spherical(R=vp.Scalar(vp.Uniform([0,Rsphere]))))
    def density(r:vp.Vector):
        return np.ones(r.shape[0])
        
    assert_integral_is_close(density, 123)

