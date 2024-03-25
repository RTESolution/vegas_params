from .base import Expression
from collections import namedtuple
from typing import Sequence
import numpy as np

    
class ParamTuple(Expression):
    """A class to pack your values to a tuple"""
    def __init__(self, name:str="ParamTuple", **parameters):
        super().__init__(**parameters)
        self.data_class = namedtuple(name, list(parameters.keys()))
    def make(self, *args:Sequence[np.ndarray]):
        return self.data_class(*args)

class ParamDict(Expression):
    """A class to pack your values to a dict"""
    def __init__(self, **parameters):
        super().__init__(**parameters)
    def make(self, **kwargs)->np.ndarray:
        return kwargs
