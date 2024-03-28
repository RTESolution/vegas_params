import vegas
from .base import Expression
from warnings import warn
import numpy as np
from loguru import logger

def integral(e: Expression):
    """decorator for turning expression into integral"""
    integrator = vegas.Integrator(e.input_limits)
    @vegas.lbatchintegrand
    def _integrand(x):
        result = e(x)
        if isinstance(result, np.ndarray):
            return np.squeeze(result) * np.squeeze(e.factor)
        elif isinstance(result, dict):
            return {key:np.squeeze(value)*np.squeeze(e.factor) for key,value in result.items()}
        else:
            logger.warn('Failed to apply the factor to result of type {}',type(result))
            return result
        
    def _run(**vegas_parameters):
        return integrator(_integrand, **vegas_parameters)
    return _run
