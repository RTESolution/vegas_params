import numpy as np
from .base import expression, Uniform, Fixed

class vector(np.ndarray):
    _vector_axis = -1
    def mag(self)->np.ndarray:
        return np.linalg.norm(self, axis=self._vector_axis, keepdims=True)
    def mag2(self)->np.ndarray:
        return self.mag()**2
    def dot(self, other)->np.ndarray:
        return np.sum(self*other, axis=self._vector_axis, keepdims=True)
    @property
    def x(self):
        return self.take(0, axis=self._vector_axis)
    @property
    def y(self):
        return self.take(1, axis=self._vector_axis)
    @property
    def z(self):
        return self.take(2, axis=self._vector_axis)
        
@expression 
class Vector:
    """Construct vector from given coordinates"""
    xyz:Uniform = Fixed([0,0,0])
    @staticmethod
    def make(xyz):
        return xyz.reshape(xyz.shape[0],-1,3).view(vector)

@expression
class Scalar:
    """Construct vector from given coordinates"""
    x:Uniform = Fixed(0)
    @staticmethod
    def make(x):
        return x.reshape(x.shape[0],-1,1).view(vector)

@expression
class Direction(Vector):
    """Generate unit vector in polar coordinates"""
    cos_theta:Uniform = Uniform([-1,1]) 
    phi:Uniform = Uniform([0,2*np.pi])
    @staticmethod
    def make(cos_theta:np.ndarray, phi:np.ndarray)->np.ndarray:
        sin_theta = np.sqrt(1-cos_theta**2)
        return np.stack([sin_theta*np.cos(phi),sin_theta*np.sin(phi), cos_theta], axis=-1).view(vector)


@expression
class PolarVector(Vector):
    R:Uniform = 1
    s:Direction = Direction()
    @staticmethod
    def make(R,s):
        return R*s