import numpy as np
import scipy.stats as sps
import fastnorm as fn
from fastnorm.util import Timer

try:
    # https://github.com/david-cortes/approxcdf
    import approxcdf as acdf

    has_approxcdf = True
except:
    has_approxcdf = False

rho = 0.5

with Timer("Scipy", decimals=6):
    sps.multivariate_normal.cdf([1, 1], mean=[0, 0], cov=[[1, rho], [rho, 1]])

with Timer("Fastnorm", decimals=6):
    fn.bivar_norm_cdf([1, 1], rho)

if has_approxcdf:
    with Timer("Approx_cdf", decimals=6):
        acdf.bvn_cdf(1, 1, rho)

x = np.random.randn(1000, 2)
rho = 0.99

with Timer("Scipy", decimals=6):
    sps.multivariate_normal.cdf(x, mean=[0, 0], cov=[[1, rho], [rho, 1]])

with Timer("Fastnorm", decimals=6):
    fn.bivar_norm_cdf(x, rho)

if has_approxcdf:
    with Timer("Approx_cdf", decimals=6):
        [acdf.bvn_cdf(xx[0], xx[1], rho) for xx in x]
