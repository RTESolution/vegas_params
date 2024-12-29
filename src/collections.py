from .base import Expression, Parameter
from collections import namedtuple
from typing import Sequence
import numpy as np

class Concat(Expression):
    """Concatenate arrays along axis=1 into a single array"""
    def __init__(self, *expressions:Sequence[Parameter]):
        super().__init__(**{f"p_{num}":expr for num,expr in enumerate(expressions)})
    @staticmethod
    def __call__(*args:Sequence[np.ndarray])->np.ndarray:
        result = np.concatenate(args,axis=1)
        return result.view(type(args[0]))
    def __or__(self, other):
        #if concatenating with another Concat
        return Concat(*list(self.parameters.values()), other)

#add the operators
Parameter.__or__ = lambda self,other: Concat(self,other)
Parameter.__ror__ = lambda self,other: Concat(other, self)


class StructArray(Expression):
    """Conctenate arrays along given axis, and form them into a np.structured_array"""
    def __init__(self, axis=-1, **parameters):
        self.axis = axis
        #initilaize as expression
        super().__init__(**parameters)
        
    def __call__(self, *args:Sequence[np.ndarray])->np.ndarray:
        #store the formats for each argument
        formats = []
        for a in args:
            size = a.shape[self.axis]
            fmt = (a.dtype, size)
            formats.append(fmt)
        dtype = np.dtype({'names':list(self.parameters),'formats':formats})
        result = np.concatenate(args, axis=self.axis)
        return result.view(dtype=dtype).squeeze()