# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2020)
#
# This file is part of PySchild.
#
# PySchild is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PySchild is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PySchild.  If not, see <http://www.gnu.org/licenses/>.

"""Simulate the response of human eyeballs to electromagnetic spectra

This module provides tools for approximating the visual color of a light
source emanating with a known intensity spectrum, particularly a perfect
blackbody of fixed temperature. The approach used here follows a tutorial
found at https://scipython.com/blog/converting-a-spectrum-to-a-colour/.
"""

import numpy

from scipy.constants import (c, h, k)
from scipy.integrate import quad

__author__ = "Alex Urban <alexander.urban@ligo.org>"

# define color system
# based on the standard Red Green Blue (sGRB), IEC 61966-2-1:1999
# for details see https://webstore.iec.ch/publication/6169
CSYS = numpy.array((
    (0.64, 0.33, 0.03),  # red
    (0.30, 0.60, 0.10),  # green
    (0.15, 0.06, 0.79),  # blue
)).T

# derive xyz -> rgb transformation matrix
CSYS_INV = numpy.linalg.inv(CSYS)
WHITE_D65 = (0.3127, 0.3290, 0.3583)
WSCALE = (CSYS_INV @ WHITE_D65)[:, numpy.newaxis]
TRANS = CSYS_INV / WSCALE


# -- tristimulus tools --------------------------------------------------------

def asymmetric_gaussian(x, alpha, mu, sig1, sig2):
    """
    """
    return numpy.piecewise(x, [x < mu, x >= mu], [
        lambda x: alpha * numpy.exp(-(x - mu)**2 / (2 * sig1**2)),
        lambda x: alpha * numpy.exp(-(x - mu)**2 / (2 * sig2**2)),
    ])


## FIXME: remember that wvlngth is in Angstroms
def xbar(wvlngth):
    """https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return (
        asymmetric_gaussian(wvlngth, 1.056, 5998, 379, 310) +
        asymmetric_gaussian(wvlngth, 0.362, 4420, 160, 267) +
        asymmetric_gaussian(wvlngth, -0.065, 5011, 204, 262)
    )


def ybar(wvlngth):
    """https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return (
        asymmetric_gaussian(wvlngth, 0.821, 5688, 469, 405) +
        asymmetric_gaussian(wvlngth, 0.286, 5309, 163, 311)
    )


def zbar(wvlngth):
    """https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return (
        asymmetric_gaussian(wvlngth, 1.217, 4370, 118, 360) +
        asymmetric_gaussian(wvlngth, 0.681, 4590, 260, 138)
    )


def rgb_from_spectrum(spec, wlmin=3800, wlmax=7800, **kwargs):
    """Spectral radiance must be defined over the wavelength in Angstroms
    """
    # integrate spectrum against color-matching functions
    (x, _) = quad(lambda wl: xbar(wl) * spec(wl, **kwargs), wlmin, wlmax)
    (y, _) = quad(lambda wl: ybar(wl) * spec(wl, **kwargs), wlmin, wlmax)
    (z, _) = quad(lambda wl: zbar(wl) * spec(wl, **kwargs), wlmin, wlmax)
    # convert to RGB color value
    xyz = numpy.array((x, y, z)) / (x + y + z)
    rgb = TRANS @ xyz
    # desaturate if outside the RGB gamut, then normalize
    rgb -= min(0, rgb.min())
    norm = rgb.max()
    return ((rgb / norm) if norm else rgb)


# -- blackbody spectra --------------------------------------------------------

def blackbody_spectrum(wvlngth, temp):
    """
    """
    # convert to SI units
    lambda_si = wvlngth * 1e-10
    factor = h * c / (lambda_si * k * temp)
    return 2 * h * c**2 / lambda_si**5 / (numpy.exp(factor) - 1)


def blackbody_color(temp):
    """
    """
    return rgb_from_spectrum(blackbody_spectrum, temp=temp)
