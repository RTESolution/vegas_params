![PyPI - Version](https://img.shields.io/pypi/v/vegas_params)
![Test](https://github.com/RTESolution/vegas_params/actions/workflows/test.yml/badge.svg)

# vegas_params
Wrapper package for [Vegas](https://vegas.readthedocs.io) integrator.

This package allows user to define composite expressions, which can be used to:
1. Calculate integrals using `vegas` algorithm
2. Make random samples of given expressions (for MC simulation apart from `vegas`)

## Installation

You can install `vegas_params` from PyPI:
```shell
pip install vegas_params
```

or directly via github link:
```shell
pip install git+https://github.com/RTESolution/vegas_params.git
```

## Quickstart
### Gaussian integral
Let's caluclate integral 
$$
\int\limits_{-1000}^{1000}dx \int\limits_{-1000}^{1000}dy \; e^{-(x^2+y^2)}
$$

Here is how you can do it:
```python
import vegas_params as vp
@vp.integral
@vp.expression(x=vp.Uniform([-1000,1000]),
            y=vp.Uniform([-1000,1000]))
def gaussian(x, y):
    return np.exp(-(x**2+y**2))
```
Now `gaussian` is a callable, where you can pass the parameters of the vegas integrator:
```python
gaussian(nitn=20, neval=100000)
#3.13805(16) - close to expected Pi
```

## Defining expressions

### Basic parameters
There are basic types of expressions:
```python
#fixed value:
c = vp.Fixed(299792458) #speed of light in vacuum in m/s
#uniform value, that will be integrated over
cos_theta = vp.Uniform([-1,1]) #cos_theta is in the given range
```

### Compound expressions
Other expressions can be defined from these components.

More complex expressions can be defined using `vp.Expression` class or `vp.expression` decorator:

```python
@vp.expression
def product(x,y):
    return x*y
    
```

It's possible to also add the Jacobian factor to the expression. 
To do this, add `self` argument to function and set `serlf.factor` to needed value. This value will be multiplied by the expression value during the integration.

## Sampling

The `vegas_params` expressions and parameters can be used not only for integration, but as random value generators.

You can ask an expression to generate a sample of given size:
```python
data = expr.sample(size=1000)
```

If your expression, or it's components (the expression parameters, provided in constructor) define the `factor`, then a simple `sample` method will not take it into account.

You need to use `sample_with_factor` method, which will iteratively generate samples and filter them, by randomly discarding the samples according to their `factor` values:
```python
data = expr.sample_with_factor(size=1000)
```
