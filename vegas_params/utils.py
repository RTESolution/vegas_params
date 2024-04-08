import numpy as np
from functools import wraps
from typing import Callable, Iterable, Mapping
import inspect

def swap_axes(a):
    if isinstance(a, np.ndarray):
        return np.swapaxes(a,0,1)
    elif isinstance(a, Mapping):
        #handle the mapping
        data = {name: swap_axes(value) for name,value in a.items()}
        return a.__class__(**data)
    elif isinstance(a, Iterable):
        #handle the iterable
        data = [swap_axes(value) for value in a]
        return a.__class__(*data)
    elif hasattr(a, "__dataclass_fields__"):
        #handle the dataclass
        data = {name: swap_axes(a.__dict__[name]) for name in a.__dataclass_fields__}
        return a.__class__(**data)
    else:
        raise TypeError(f"Cannot swap axes for type {type(a)}")

def mask_array(a:np.ndarray, mask:np.ndarray, axis=1)->np.ndarray:
    """produce a smaller array (where mask==False)"""
    result = a.take(np.flatnonzero(~mask),axis=axis)
    return result
   
def unmask_array(a:np.ndarray, mask:np.ndarray, fill_value=0, axis=1)->np.ndarray:
    """produce a larger array, filling the values where `mask==True` with the `fill_value`"""
    shape = list(a.shape)
    shape[axis] = len(mask)
    result = np.ones(shape, dtype=a.dtype)*fill_value
    result[:,~mask] = a
    return result

def swapaxes(func):
    S = inspect.signature(func)
    is_static = list(S.parameters)[0]!='self'
    
    @wraps(func)
    def _f(*args,**kwargs):
        params = S.bind(*args, **kwargs)
        params.apply_defaults()
        if is_static:
            args_T = swap_axes(params.args)
        else:
            #swap all arguments except 'self'
            args_T = [params.args[0],*swap_axes(params.args[1:])]
        kwargs_T = swap_axes(params.kwargs)
        #pass as positional arguments 
        return func(*args_T, **kwargs_T)
    return _f

def mask_all(a:Iterable[np.ndarray], mask:np.ndarray, axis=1)->Iterable[np.ndarray]:
    """produce a smaller array (where mask==False) - for all arrays in given iterable (works on NamedTuples too)"""
    return a.__class__(*[mask_array(f, mask, axis) for f in a])