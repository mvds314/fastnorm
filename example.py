import scipy.stats as sps
import fastnorm as fn
from fastnorm.util import Timer

rho = 0.5

with Timer("Scipy", decimals=6):
    sps.multivariate_normal.cdf([1, 1], mean=[0, 0], cov=[[1, rho], [rho, 1]])

with Timer("Fastnorm", decimals=6):
    fn.bivar_norm_cdf([1, 1], rho)
