import vegas
from .base import Expression
from warnings import warn
import numpy as np

def integral(e: Expression):
    """decorator for turning expression into integral"""
    integrator = vegas.Integrator(e.input_limits)
    @vegas.lbatchintegrand
    def _integrand(x):
        result = e.__construct__(x)
        if isinstance(result, dict):
            return {key:np.squeeze(value)*np.squeeze(e.factor) for key,value in result.items()}
        else:
            return np.squeeze(result) * np.squeeze(e.factor)
        
    def _run(**vegas_parameters):
        return integrator(_integrand, **vegas_parameters)
    return _run
