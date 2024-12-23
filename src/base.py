import numpy as np
import textwrap
import abc
from typing import Callable, Sequence
import inspect
from functools import wraps

class Parameter(abc.ABC):
    """Base class for the parameter in the expression"""
    factor = 1
    def __init__(self):
        self.input_limits = None
    def __len__(self):
        return self.input_limits.shape[0]
    
    @abc.abstractmethod
    def __construct__(self, x:np.ndarray)->np.ndarray:
        """Calculate this parameter value from the input array"""
    def sample(self, size=1):
        """Generate a random sample of this parameter values"""
        x = np.random.uniform(self.input_limits[:,0],
                              self.input_limits[:,1], 
                              size=[size,len(self)])
        values = self.__construct__(x)
        #self.factor*=np.ones(size) #make sure the factor is array of size len(sample)
        return values
        
    def sample_with_factor(self, size=1, iter_max=10):
        """Generate a random sample of this expression values, 
        taking into account the factor as the probability density.

        Note: this process is iterative (it might need to generate several samples), and several times slower than regular :meth:`sample` method
        """
        #start with the sample of size N
        the_sample = self.sample(0) #just get the output size
        the_factor = self.factor*np.ones(shape=(0,))
        the_random = np.empty_like(the_factor)
        selected = []
        N = size
        N_generate = int(N*2) #the size of next sample to generate
        for it in range(iter_max):
            sample, factor = self.sample(N_generate), self.factor
            #expand factor
            factor = factor*np.ones(shape=(N_generate))
            if it==0:
                if np.min(self.factor)==np.max(self.factor):
                    #the factor is equal for all values, so just return this sample
                    return sample[:N]
                    
            random = np.random.uniform(size=len(sample))
            the_sample = np.append(the_sample, sample, axis=0)
            the_factor = np.append(the_factor, factor, axis=0)
            the_random = np.append(the_random, random, axis=0)
            #apply the selection
            selected = (the_random*the_factor.max()) < the_factor
            N_selected = selected.sum()
            print(f"Iteration #{it}: generated {N_generate} -> selected={N_selected}/{len(the_factor)}")
            if N_selected>=N:
                return the_sample[selected][:N]
            else:
                prob = N_selected/len(the_sample) #estimated selection probability
                #now decide how many items to generate
                # N = prob*(N_total) = N_selected + prob*N_generate =>
                N_generate = (N-N_selected)/prob
                #additional 20% to avoid making small iterations
                N_generate *= 1.2
                #make sure we don't make a sample too small or too large
                N_generate = np.clip(N_generate, N/100, N*1000)
                N_generate = np.asarray(np.ceil(N_generate), dtype=int)
        raise RuntimeError(f"Maximum iterations ({iter_max}) reached. Generated only {sum(selected)} points of {N} requested")


class Fixed(Parameter):
    """Fixed value, not a part of integration"""
    def __init__(self, value=0):
        self.input_limits = np.empty(shape=(0,2))
        self.value = np.atleast_1d(value)[np.newaxis,:]
    def __construct__(self, x):
        shape = (x.shape[0],*self.value.shape[1:])
        result = np.ones(shape)*self.value
        return result
    def __pow__(self, n:int)->'Fixed':
        return self.__class__(value=np.repeat(self.value, n, axis=0))
    def __repr__(self):
        return f'{self.__class__.__name__}[{len(self)}]({self.value})'

class Uniform(Parameter):
    """Parameter integrated in the given limits"""
    def __init__(self, limits=[0,1]):
        self.input_limits = np.array(limits, ndmin=2)
    def __construct__(self, x:np.ndarray)->np.ndarray:
        return x
    def __pow__(self, n:int)->'Uniform':
        return self.__class__(limits=np.repeat(self.input_limits, n, axis=0))
    def __repr__(self):
        label =  f'{self.__class__.__name__}[{len(self)}]'
        limits = ','.join(str(list(a)) for a in self.input_limits)
        return f'{label}({limits})'

def _make_parameter(x):
    if isinstance(x, Parameter):
        return x
    else:
        return Fixed(x)
        
class Expression(Uniform):
    """Complex expression of the given input parameters"""
    def __init__(self, **parameters):
        self.parameters = {name:_make_parameter(par) for name,par in  parameters.items()} 
    def __getitem__(self, name:str)->Parameter:
        return self.parameters[name]
    def __setitem__(self, name:str, value):
        self.parameters[name] = _make_parameter(value)
        
    @property
    def input_limits(self):
        return np.concatenate([par.input_limits for par in self.parameters.values()])

    def __construct__(self, x:np.ndarray):
        #evaluate all the parameters 
        n=0
        par_values = {}
        factor_from_parameters = 1
        for name,par in self.parameters.items():
            par_values[name] = par.__construct__(x[:,n:n+len(par)])
            factor_from_parameters *= par.factor
            n+=len(par)
        #run the final evaluation function
        self.factor = 1
        result = self.__call__(*par_values.values())
        #calculate the resulting factor
        self.factor = self.factor * factor_from_parameters
        self.factor = np.squeeze(self.factor)
        return result
        
    def __call__(self, *args):
        """Calculate the expression value(s) based on the input values"""
        return dict(zip(self.parameters,args))
    def __pow__(self, n:int)->'Expression':
        return self.__class__(**{name: par**n for name,par in self.parameters.items()})
    def __repr__(self):
    #    return f'{self.__class__.__name__}[{len(self)}]('+'\n,'.join([f'{name}={repr(par)}' for name,par in self.parameters.items()])+')'
    #def __str__(self):
        label = f'{self.__class__.__name__}[{len(self)}]'
        pars = '\n'.join([f' --> {name}={repr(par)}' for name,par in self.parameters.items()])
        return f'{label}(\n{textwrap.indent(pars,"    ")}\n)'

class Concat(Expression):
    def __init__(self, *expressions:Sequence[Parameter]):
        super().__init__(**{f"p_{num}":expr for num,expr in enumerate(expressions)})
    @staticmethod
    def __call__(*args:Sequence[np.ndarray])->np.ndarray:
        result = np.concatenate(args,axis=1)
        return result.view(type(args[0]))
    def __or__(self, other):
        #if concatenating with another Concat
        return Concat(*list(self.parameters.values()), other)

class FromDistribution(Expression):
    """Sample the input from the given distribution"""
    def __init__(self, ppf:callable, size:int=1):
        """Parameters:
        ppf: Callable
            Percent point function (inverse of cumulative density function — percentiles)
        """
        self.ppf = ppf
        super().__init__(y=Uniform([0,1])**size)
    def __call__(self,y):
        return self.ppf(y)
    def __pow__(self, n:int):
        return self.__class__(self.ppf, len(self)*n)


def _expression_from_class(c):
    """A class decorator to make an Expression class"""
    #check the signature of the callable
    S = inspect.signature(c.__call__)
    arguments = list(S.parameters)
    if arguments[0]=='self':
        del arguments[0]    
    #prepare the signature
    params = [inspect.Parameter(name=name, 
                                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=c.__dict__.get(name, S.parameters[name].default),
                                annotation=c.__annotations__.get(name, S.parameters[name].annotation),
                               ) for name in arguments]
    S = inspect.Signature(parameters=params)
    #create a new class
    #define the base class
    if issubclass(c,Expression):
        bases = [c]
    else:
        bases = (c,Expression)
    class c1(*bases):
        def __init__(self, *args, **kwargs):
            params = S.bind(*args, **kwargs)
            params.apply_defaults()
            Expression.__init__(self, **params.arguments)
    c1.__signature__ = S
    return c1

def expression(obj=None, **parameters):
    """A decorator to create expression classes from functions or classes."""
    if obj is None:
        #no object (class or callable) was passed, so let's delegate this to the next layer
        def _make_expression(the_obj):
            return expression(the_obj, **parameters)
        return _make_expression
        
    if inspect.isclass(obj):
        cls = _expression_from_class(obj)
        
    elif isinstance(obj, Callable):
        #make a class and wrap it
        S = inspect.signature(obj)
        arguments = list(S.parameters)
        if arguments[0]=='self':
            del arguments[0]
            make_function = obj
        else:
            make_function = staticmethod(obj)
        class C:
            #check if a first argument is "self"
            __call__=make_function
        cls = _expression_from_class(C)
    else:
        raise TypeError(f"`obj` must be a class or a function, not a {obj.__class__.__name__}")

    #fill the class and module name to be the same as in class
    cls.__qualname__ = obj.__qualname__
    cls.__name__ = obj.__name__
    cls.__module__ = obj.__module__
    cls.__doc__ = obj.__doc__
    if parameters=={}:
        return cls
    else:
        return cls(**parameters)

def forward_input(func):
    """A function decorator to return the dict of input arguments alongside with the function result"""
    S = inspect.signature(func)
    @wraps(func)
    def _f(*args, **kwargs):
        #get the input dict
        arguments = S.bind(*args, **kwargs).arguments
        #get the function result
        result = func(**arguments)
        return result, arguments
    return _f
    
#additional methods
Shifted = expression(lambda v,shift: v+shift)
Scaled = expression(lambda v,factor: v*factor)

#add the operators
Parameter.__or__ = lambda self,other: Concat(self,other)
Parameter.__ror__ = lambda self,other: Concat(other, self)
Parameter.__add__ = lambda self,other: Shifted(self,other)
Parameter.__radd__ = lambda self,other: Shifted(other, self)
Parameter.__mul__ = lambda x,y:Scaled(x,y)
Parameter.__rmul__ = lambda x,y:Scaled(y,x)