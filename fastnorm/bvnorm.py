# -*- coding: utf-8 -*-
"""
Some code for the bivariate normal distribution

The code is based on the code by Alan Genz, which is available at
http://www.math.wsu.edu/faculty/genz/software/tvn.m

Copyright (C) 2011, Alan Genz,  All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided the following conditions are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. The contributor name(s) may not be used to endorse or promote
     products derived from this software without specific prior
     written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import math
import scipy.stats as sps

from .util import type_wrapper


@type_wrapper(xloc=0)
def bivar_norm_pdf(x, rho):
    r"""
    Evaluate the bivariate (standard) normal distribution function with correlation :math:`\rho`

    .. math::
        f([x_0,x_1],\rho)=\frac{e^{-\frac{x_0^2+x_1^2-2\rho x_0 x_1}{2(1-\rho ^2)}}}{2 \pi  \sqrt{1-\rho ^2}}
    """
    if len(x) != 2:
        raise Exception("Length of x is assumed to be 2")
    if np.abs(rho) >= 1:
        raise Exception("rho should be between -1 and 1")
    if np.any(np.isinf(x)):
        return 0
    else:
        return (
            1
            / (2 * math.pi * math.sqrt(1 - rho**2))
            * math.exp(-1 * (x[0] ** 2 + x[1] ** 2 - 2 * rho * x[0] * x[1]) / 2 / (1 - rho**2))
        )


@type_wrapper(xloc=0)
def bivar_norm_cdf(a, b, rho=0):
    r"""
    Evaluate bivariate cummulative standard normal distribution function

    .. math::
        \int_{-\infty}^{a} \int_{-\infty}^{b} \phi(x,y,\rho) dxdy

    with correlation coefficient :math:`\rho`.

    This function is based on the method described by
    Drezner, Z and G.O. Wesolowsky, (1989), On the computation of the bivariate normal inegral, Journal of Statist. Comput. Simul. 35, pp. 101-107.
    """
    return bvnl(a, b, rho)


_w_bvnu_1 = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
_x_bvnu_1 = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
_w_bvnu_2 = np.array(
    [
        0.04717533638651177,
        0.1069393259953183,
        0.1600783285433464,
        0.2031674267230659,
        0.2334925365383547,
        0.2491470458134029,
    ]
)
_x_bvnu_2 = np.array(
    [
        0.9815606342467191,
        0.9041172563704750,
        0.7699026741943050,
        0.5873179542866171,
        0.3678314989981802,
        0.1252334085114692,
    ]
)
_w_bvnu_3 = np.array(
    [
        0.01761400713915212,
        0.04060142980038694,
        0.06267204833410906,
        0.08327674157670475,
        0.1019301198172404,
        0.1181945319615184,
        0.1316886384491766,
        0.1420961093183821,
        0.1491729864726037,
        0.1527533871307259,
    ]
)
_x_bvnu_3 = np.array(
    [
        0.9931285991850949,
        0.9639719272779138,
        0.9122344282513259,
        0.8391169718222188,
        0.7463319064601508,
        0.6360536807265150,
        0.5108670019508271,
        0.3737060887154196,
        0.2277858511416451,
        0.07652652113349733,
    ]
)
_w_bvnu_1 = np.hstack((_w_bvnu_1, _w_bvnu_1))
_x_bvnu_1 = np.hstack((1 - _x_bvnu_1, 1 + _x_bvnu_1))
_w_bvnu_2 = np.hstack((_w_bvnu_2, _w_bvnu_2))
_x_bvnu_2 = np.hstack((1 - _x_bvnu_2, 1 + _x_bvnu_2))
_w_bvnu_3 = np.hstack((_w_bvnu_3, _w_bvnu_3))
_x_bvnu_3 = np.hstack((1 - _x_bvnu_3, 1 + _x_bvnu_3))


def bvnu(a, b, rho=0):
    r"""
    Evaluate bivariate cummulative standard normal distribution function

    .. math::
        \int_a^\infty \int_b^\infty \phi(x,y,\rho) dxdy

    with correlation coefficient :math:`\rho`.

    This function is based on the method described in [1].

    References
    ----------
    .. [1] Drezner, Z and G.O. Wesolowsky, (1989), On the computation of the bivariate normal inegral, Journal of Statist. Comput. Simul. 35, pp. 101-107.
    """
    global _w_bvnu_1, _x_bvnu_1, _w_bvnu_2, _x_bvnu_2, _w_bvnu_3, _x_bvnu_3
    if np.isposinf(a) or np.isposinf(b):
        p = 0
    elif np.isneginf(a):
        if np.isneginf(b):
            p = 1
        else:
            p = sps.norm.cdf(-b)
    elif np.isneginf(b):
        p = sps.norm.cdf(-a)
    elif rho == 0:
        p = sps.norm.cdf(-a) * sps.norm.cdf(-b)
    else:
        tp = 2 * math.pi
        h = a
        k = b
        hk = h * k
        bvn = 0
        if abs(rho) < 0.3:  # Gauss Legendre points and weights, n =  6
            w = _w_bvnu_1
            x = _x_bvnu_1
        elif abs(rho) < 0.75:  # Gauss Legendre points and weights, n = 12
            w = _w_bvnu_2
            x = _x_bvnu_2
        else:  # Gauss Legendre points and weights, n = 20
            w = _w_bvnu_3
            x = _x_bvnu_3
        if abs(rho) < 0.925:
            hs = (h * h + k * k) / 2
            asr = math.asin(rho) / 2
            sn = np.sin(asr * x)
            bvn = np.dot(np.exp((sn * hk - hs) / (1 - sn**2)), w)
            bvn = bvn * asr / tp + sps.norm.cdf(-h) * sps.norm.cdf(-k)
        else:
            if rho < 0:
                k = -k
                hk = -hk
            if abs(rho) < 1:
                ass = 1 - rho**2
                a = math.sqrt(ass)
                bs = (h - k) ** 2
                asr = -(bs / ass + hk) / 2
                c = (4 - hk) / 8
                d = (12 - hk) / 80
                if asr > -100:
                    bvn = (
                        a
                        * math.exp(asr)
                        * (1 - c * (bs - ass) * (1 - d * bs) / 3 + c * d * ass**2)
                    )
                if hk > -100:
                    b = math.sqrt(bs)
                    spp = math.sqrt(tp) * sps.norm.cdf(-b / a)
                    bvn = bvn - math.exp(-hk / 2) * spp * b * (1 - c * bs * (1 - d * bs) / 3)
                a = a / 2
                xs = (a * x) ** 2
                asr = -(bs / xs + hk) / 2
                ix = asr > -100
                xs = xs[ix]
                spp = 1 + c * xs * (1 + 5 * d * xs)
                rs = np.sqrt(1 - xs)
                ep = np.exp(-(hk / 2) * xs / (1 + rs) ** 2) / rs
                bvn = (a * np.dot(np.exp(asr[ix]) * (spp - ep), w[ix]) - bvn) / tp
            if rho > 0:
                bvn = bvn + sps.norm.cdf(-max(h, k))
            elif h >= k:
                bvn = -bvn
            else:
                if h < 0:
                    L = sps.norm.cdf(k) - sps.norm.cdf(h)
                else:
                    L = sps.norm.cdf(-h) - sps.norm.cdf(-k)
                bvn = L - bvn
        p = max(0, min(1, bvn))
    return p


def bvnl(a, b, rho=0):
    r"""
    Evaluate bivariate cummulative standard normal distribution function

    .. math::
        \int_{-\infty}^{a} \int_{\infty}^{b} \phi(x,y,\rho) dxdy

    with correlation coefficient :math:`\rho`.
    """
    return bvnu(-a, -b, rho)