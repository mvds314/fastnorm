#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import numpy as np
import scipy.stats as sps

from fastnorm import bivar_norm_pdf, bivar_norm_cdf
from fastnorm.bvnorm import bvnu


@pytest.mark.parametrize(
    "rho,xys", [(0, [[0, 0], [1, 1], [2, 2]]), (0.5, [[0, 0], [1, 1], [2, 2]])]
)
def test_bivar_norm_pdf(rho, xys):
    for xy in xys:
        assert np.isclose(
            bivar_norm_pdf(xy, rho),
            sps.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).pdf(xy),
        )
    assert np.allclose(
        bivar_norm_pdf(xys, rho),
        sps.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).pdf(xys),
    )


@pytest.mark.parametrize(
    "rho,xys",
    [
        (0, [[0, 0], [1, 1], [2, 2]]),
        (0.5, [[0, 0], [1, 1], [2, 2]]),
        (-0.615145406302, [[1, 2], [2, 3], [3, 4]]),
        (0.1, [[1, 2], [2, 3], [3, 4]]),
        (0.99, [[1, 2], [2, 3], [3, 4]]),
        (-0.86, [[1, 2], [2, 3], [3, 4]]),
        (-0.99, [[1, 2], [2, 3], [3, 4]]),
        (0.3, [[1, 2], [2, 3], [3, 4]]),
        (0.3, [[1, np.inf], [2, np.inf], [3, np.inf]]),
    ],
)
def test_bivar_norm_cdf(rho, xys):
    for xy in xys:
        assert np.isclose(
            bivar_norm_cdf(xy, rho),
            sps.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(xy),
        )
    assert np.allclose(
        bivar_norm_cdf(xys, rho),
        sps.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(xys),
    )


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_bivar_norm_cdf",
                "--tb=auto",
                "--pdb",
            ]
        )
