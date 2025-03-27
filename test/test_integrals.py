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

def test_integral_with_external_adapt():
    #given a large space
    space_xy = vp.Vector(vp.Uniform([-1000,1000])**2)
    #find an area of a tiny circle
    center = vp.Vector([5,1])
    radius=1e-1
    @vp.expression(R=space_xy, center=center, radius=vp.Scalar(radius))
    def sharp_function(R, center, radius):
        """an expression which is nonzero in a very tiny region around position"""
        dr = np.linalg.norm(R-center, axis=-1, keepdims=True)
        return 1.*(dr < radius).squeeze()

    @vp.expression(R=space_xy, center=center, radius=vp.Scalar(radius))
    def soft_function(R, center, radius):
        """a softer function, which is nonzero everywhere"""
        dr = np.linalg.norm(R-center, axis=-1, keepdims=True)
        return np.exp(-dr/radius).squeeze()

    #check that simple integral with standard adaptation gives us 0
    i_default = vp.integral(sharp_function)(nitn=50, neval=10000, adapt=True)
    assert np.isclose(i_default.mean, 0)

    #check that simple integral with standard adaptation gives us 0
    i_adapt = vp.integral(sharp_function)(nitn=50, neval=10000, adapt=soft_function)
    I_expected = np.pi*radius**2
    assert np.isclose(i_adapt.mean, I_expected, rtol=1e-1)
    
    