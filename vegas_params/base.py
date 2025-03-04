import numpy as np
import textwrap
import abc
from typing import Callable,Tuple,List, Sequence
from collections import namedtuple
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
        pass
    def sample(self, size=1):
        """Generate a random sample of this parameter values"""
        x = np.random.uniform(self.input_limits[:,0],
                              self.input_limits[:,1], 
                              size=[size,len(self)])
        return self.__construct__(x)

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

#define the decorators
def expression_from_callable(**parameters):
    """decorator for creating the Expression from function"""
    def _wrapper(obj):
        c = Expression(**parameters)
        c.__name__ = obj.__name__
        c.__qualname__ = obj.__qualname__
        c.__call__ = obj
        return c
    return _wrapper

def expression_from_class(c):
    """A class decorator to make an Expression class"""
    #prepare the signature
    params = [inspect.Parameter(name=name, 
                                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=c.__dict__.get(name, inspect.Parameter.empty),
                                annotation=annot if annot!=Parameter else inspect.Parameter.empty
                               ) for name,annot in c.__annotations__.items()]
    
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
    #fill the class and module name to be the same as in class
    c1.__qualname__ = c.__qualname__
    c1.__name__ = c.__name__
    c1.__module__ = c.__module__
    c1.__signature__ = S
    return c1

def expression(obj=None, **parameters):
    """A decorator to create expression classes from functions or classes."""
    if obj is None:
        return expression_from_callable(**parameters)
    
    if inspect.isclass(obj):
        return expression_from_class(obj)
    elif isinstance(obj, Callable):
        #make a class and wrap it
        S = inspect.signature(obj)
        parameters = list(S.parameters)
        if parameters[0]=='self':
            del parameters[0]
            make_function = obj
        else:
            make_function = staticmethod(obj)
        class c:
            #check if a first argument is "self"
            __call__=make_function
        #set parameters
            
        c.__annotations__ = {name:Parameter for name in parameters}
        c.__qualname__ = obj.__qualname__
        c.__doc__ = obj.__doc__
        c.__name__ = obj.__name__
        c.__module__ = obj.__module__
        return expression_from_class(c)
    else:
        raise TypeError(f"`obj` must be a class or a function, not a {obj.__class__.__name__}")

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
@expression
def Shifted(v,shift):
    return v+shift
    
@expression
def Scaled(v,factor):
    return v*factor

#add the operators
Parameter.__or__ = lambda self,other: Concat(self,other)
Parameter.__ror__ = lambda self,other: Concat(other, self)
Parameter.__add__ = lambda self,other: Shifted(self,other)
Parameter.__radd__ = lambda self,other: Shifted(other, self)
Parameter.__mul__ = lambda x,y:Scaled(x,y)
Parameter.__rmul__ = lambda x,y:Scaled(y,x)