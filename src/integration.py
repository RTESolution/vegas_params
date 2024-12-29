import vegas
from .base import Expression
import numpy as np
from gvar import gvar
   
def integral_naiive(e: Expression):
    """naiive MC integration without vegas"""
    def _run_integral(nitn=10, neval=1000, **kwargs):
        v,w = e.sample(neval*nitn), e.factor
        #reshape for each iteration to be a in a separate row
        v = v.reshape((neval,nitn, *v.shape[1:]))
        w = w.reshape((neval,nitn, *w.shape[1:]))
        
        wv = np.squeeze(v)*np.squeeze(w)
        
        result = np.prod(np.diff(e.input_limits)) * np.sum(wv, axis=0)/v.shape[0]
        return gvar.gvar(result.mean(axis=0),result.std(axis=0))

    
def integral(e: Expression):
    """decorator for turning expression into integral"""
    def _integrand(x):
        result = e.__construct__(x)
        if isinstance(result, dict):
            return {key:np.squeeze(value)*np.squeeze(e.factor) for key,value in result.items()}
        else:
            return np.squeeze(result) * np.squeeze(e.factor)

    if(len(e)>0):
        integrator = vegas.Integrator(e.input_limits)
        def _run_integral(adapt=False, **vegas_parameters):
            if adapt:
                #run the calculation without storing the result
                integrator(vegas.lbatchintegrand(_integrand), nitn=10, neval=1000)
            return integrator(vegas.lbatchintegrand(_integrand), **vegas_parameters, adapt=False)
        return _run_integral
    
    else:    
        def _just_calculate(**vegas_parameters):
            return gvar(_integrand(np.empty(shape=(1,0))))
        
        return _just_calculate
        
    