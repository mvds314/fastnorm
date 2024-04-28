import numpy as np
import scipy.stats as sps

import fastnorm as fn
from fastnorm.util import Timer

try:
    # https://github.com/david-cortes/approxcdf
    import approxcdf as acdf

    has_approxcdf = True
except Exception:
    has_approxcdf = False

n_tries = 100
print(f"Timing bivariate normal cdf with {n_tries} tries")

rho = 0.95
x = [1, 1]
with Timer("Scipy", decimals=6):
    for i in range(n_tries):
        sps.multivariate_normal.cdf(x, mean=[0, 0], cov=[[1, rho], [rho, 1]])

with Timer("Fastnorm", decimals=6):
    for i in range(n_tries):
        fn.bivar_norm_cdf(x, rho)

if has_approxcdf:
    with Timer("Approx_cdf", decimals=6):
        for i in range(n_tries):
            acdf.bvn_cdf(*x, rho)

x = np.random.randn(10000, 2)

with Timer("Scipy", decimals=6):
    for i in range(n_tries):
        sps.multivariate_normal.cdf(x, mean=[0, 0], cov=[[1, rho], [rho, 1]])

with Timer("Fastnorm", decimals=6):
    for i in range(n_tries):
        fn.bivar_norm_cdf(x, rho)

if has_approxcdf:
    with Timer("Approx_cdf", decimals=6):
        for i in range(n_tries):
            [acdf.bvn_cdf(xx[0], xx[1], rho) for xx in x]
