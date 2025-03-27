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

def make_integrand(e: Expression):
    def _integrand(x):
            result = e.__construct__(x)
            if isinstance(result, dict):
                return {key:np.squeeze(value)*np.squeeze(e.factor) for key,value in result.items()}
            else:
                return np.squeeze(result) * np.squeeze(e.factor)
    return _integrand
    
def integral(e: Expression):
    """decorator for turning expression into integral function"""
    if(len(e)>0):
        integrator = vegas.Integrator(e.input_limits)
        def _run_integral(adapt=False, **vegas_parameters):
            """Run the integration calculation. 
            Parameters:
            -----------
            adapt: bool or Expression
                Defines how to make vegas "adaptation" run - letting vegas to study the integrable function.
                If adapt=False (default) - no adaptation run
                If adapt=True - make adaptation run on this expression
                If adapt is Expression - use it for adaptation run. 
                This allows to run adaptation on a smoother variant of a function.
            **vegas_parameters: dict
                Keyword parameters for the vegas.Integrator 
                (see https://vegas.readthedocs.io/en/latest/vegas.html#vegas.Integrator)
            """
            if adapt==True:
                #adapt on the same function
                adapt=e
            if adapt!=False:
                #run the calculation without storing the result
                assert isinstance(adapt, Expression), f"adapt must be an Expression object, not {type(adapt)}"
                integrator(vegas.lbatchintegrand(make_integrand(adapt)), nitn=10, neval=1000)
            return integrator(vegas.lbatchintegrand(make_integrand(e)), **vegas_parameters, adapt=False)
        return _run_integral
    
    else:    
        def _just_calculate(**vegas_parameters):
            return gvar(make_integrand(e)(np.empty(shape=(1,0))))
        
        return _just_calculate
        
    