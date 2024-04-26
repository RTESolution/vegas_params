# vegas_params
Wrapper package for [Vegas](https://vegas.readthedocs.io) integrator.

## Installation

You can install `vegas_params` via github link
```shell
pip install git+https://github.com/RTESolution/vegas_params.git
```

## Examples
### Gaussian integral
Let's caluclate integral 
$$
\int\limits_{-1000}^{1000}dx \int\limits_{-1000}^{1000}dy \; e^{-(x^2+y^2)}
$$

Here is how you can do it:
```python
@integral   
@expression(x=Uniform([-1000,1000]),
            y=Uniform([-1000,1000]))
def gaussian(x, y):
    return np.exp(-(x**2+y**2))
```
Now `gaussian` is a callable, where you can pass the parameters of the vegas integrator:
```python
gaussian(nitn=20, neval=100000)
#3.13805(16) - close to expected Pi
```
