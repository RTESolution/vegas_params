import vegas
from .base import Expression

def integral(e: Expression):
    """decorator for turning expression into integral"""
    integrator = vegas.Integrator(e.input_limits)
    def _run(**vegas_parameters):
        return integrator(vegas.lbatchintegrand(e), **vegas_parameters)
    return _run
