# Fast evaluation of multivariate normal distributions

This package provides a fast way to evaluate the pdf and cdf of the multivariate normal distribution.
Currently, it only contains code for the bivariate normal distribution.

The default implementation in scipy can be slow. The implementation in this package is based on the following references:

1. Drezner, Zvi, and George O. Wesolowsky. "On the computation of the bivariate normal integral." Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107.
2. Genz, Alan, and Frank Bretz. "Computation of multivariate normal and t probabilities." Lecture Notes in Statistics 195 (2009)

Put simply, the method comes to an interpolation specifically tailored to the multivariate normal distribution.
Although it is an approximation, the method is near to exact and very fast.

## Related software

- [approxcdf](https://github.com/david-cortes/approxcdf)
- [Matlab implementations](https://www.math.wsu.edu/faculty/genz/software/software.html)

## Basic example

```python
import numpy as np
```

## Installation

You can install this library directly from github:

```bash
pip install git+https://github.com/mvds314/fastnorm.git
```

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/fastnorm.git
```

Then navigate to the folder containing `setup.py` and run

```bash
pip install -e .
```

to install the package in edit mode.

Run unittests with `pytest`.
