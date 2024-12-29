import numpy as np
from .base import expression, Uniform, Fixed

class vector(np.ndarray):
    _vector_axis = -1
    def mag(self)->np.ndarray:
        return np.linalg.norm(self, axis=self._vector_axis, keepdims=True).view(np.ndarray)
    def mag2(self)->np.ndarray:
        return self.mag()**2
    def dot(self, other)->np.ndarray:
        return np.sum(self*other, axis=self._vector_axis, keepdims=True).view(np.ndarray)
    def __mul__(self, other:np.ndarray):
        #check the dimension
        a = np.asarray(other)
        if(a.ndim==self.ndim-1):
            #expand the dimensions
            a = np.expand_dims(a, self._vector_axis)
        return super().__mul__(a)
    def __rmul__(self, other:np.ndarray):
        return self.__mul__(other)

    @property
    def x(self):
        return self.take(0, axis=self._vector_axis).view(np.ndarray)
    @property
    def y(self):
        return self.take(1, axis=self._vector_axis).view(np.ndarray)
    @property
    def z(self):
        return self.take(2, axis=self._vector_axis).view(np.ndarray)

@expression 
class Vector:
    """Construct vector from given coordinates"""
    xyz:Uniform = Fixed([0,0,0])
    @staticmethod
    def __call__(xyz):
        return xyz.reshape(xyz.shape[0],-1,xyz.shape[-1]).view(vector)

@expression
class Scalar:
    """Construct vector from given coordinates"""
    x:Uniform = Fixed(0)
    @staticmethod
    def __call__(x):
        return x.reshape(x.shape[0],-1,1)

@expression
class Direction(Vector):
    """Generate unit vector in polar coordinates"""
    cos_theta:Uniform = Uniform([-1,1]) 
    phi:Uniform = Uniform([0,2*np.pi])
    @staticmethod
    def __call__(cos_theta:np.ndarray, phi:np.ndarray)->np.ndarray:
        sin_theta = np.sqrt(1-cos_theta**2)
        return np.stack([sin_theta*np.cos(phi),sin_theta*np.sin(phi), cos_theta], axis=-1).view(vector)