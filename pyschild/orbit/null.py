# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2020)
#
# This file is part of the PySchild python package.
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

"""Core utilities for tracing light rays (i.e., null geodesics)
"""

import numpy
import warnings

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

__author__ = "Alex Urban <alexander.urban@ligo.org>"

ARCMIN = numpy.deg2rad(1) / 60


# -- utilities ----------------------------------------------------------------

def radial_potential(r, h):
    """Effective radial potential for photon orbits

    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``h`` and all terms of the potential are unitless.

    Parameters
    ----------
    r : `float` or `array-like`
        unitless radial coordinate

    h : `float`
        unitless specific orbital angular momentum

    Returns
    -------
    potential : `float` or `~numpy.ndarray`
        effective radial potential at the points represented by ``r``

    Examples
    --------
    To put this quantity in physical units, it is recommended to use ``G``,
    ``M_sun``, and ``c`` from `~astropy.constants` which explicitly carry
    units, e.g.

    >>> import numpy
    >>> from astropy.constants import (G, M_sun, c)
    >>> from pyschild.orbit.null import radial_potential
    >>> r = numpy.arange(20)
    >>> potential = radial_potential(r, 12)
    >>> r *= G * M_sun / c**2
    >>> potential *= c**2

    See also
    --------
    pyschild.orbit.timelike.radial_force
        effective radial potential for timelike geodesics
    """
    return (h/r)**2 * (1 - 2 / r) / 2


def impact_parameter(r, delta):
    """Compute the impact parameter of a photon orbit

    Parameters
    ----------
    r : `float`
        unitless radial coordinate

    delta : `float` or `array_like`
        angle (radians) between the radial direction and the
        instantaneous 4-velocity of a photon

    Returns
    -------
    b : `float` or `~numpy.ndarray`
        impact parameter correspending to these initial conditions,
        scalar if ``delta`` is scalar else an array the same size
        as ``delta``
    """
    sqrtr = numpy.sqrt(r / 2)
    num = r * sqrtr * numpy.sin(delta)
    denom = sqrtr - numpy.cos(delta)
    return num / denom


def closest_approach(b):
    """Radial coordinate of closest approach for a photon orbit

    Parameters
    ----------
    b : `float`
        impact parameter of the photon orbit

    Returns
    -------
    r0 : `float`
        radial coordinate of closest approach, will be zero if
        ``b < 3 * 3**(1/2)``
    """
    if b < numpy.sqrt(27):
        return 0
    # find the appropriate polynomial root
    roots = numpy.roots([1, 0, -b**2, 2*b**2])
    return roots[1]


def critical_angle(r):
    """Minimum local azimuthal angle to receive inbound photons

    This utility essentially determines the azimuthal size of a black hole
    relative to a spacetime event at fixed radial coordinate ``r``
    """
    num = r * numpy.sqrt(r / 2) - (r - 3) * numpy.sqrt(3 + r / 2)
    denom = numpy.sqrt(27) * (1 + numpy.sqrt(r / 2))
    return numpy.pi - 2 * numpy.arctan2(denom, num)


def integrand(r, b):
    """Integrand to find angular deflection along null geodesics

    This is a convenience function to enable numerical solutions of
    Schwarzschild orbits given a value for ``r`` and ``b`` at the same
    point(s). It uses geometrized units (``G = c = 1``) and is written
    in terms of ``r / M``, so all physical quantities are unitless.

    Parameters
    ----------
    r : `float`
        unitless radial coordinate

    b : `float`
        impact parameter along a fixed null geodesic

    Returns
    -------
    dphi_dr : `float`
        integrand evaluated at the given point with given parameter

    Notes
    -----
    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``b`` and all terms of the potential are unitless.

    The integrand here accumulates a minus sign because it is used to
    trace photon trajectories from a given spacetime event back to past
    null infinity.

    See also
    --------
    scipy.integrate.quad
        utility which numerically integrates this function at fixed
        ``delta``
    """
    denom = r * numpy.sqrt((r/b)**2 - (1 - 2/r))
    return (-1 / denom)


def far_azimuth(r, delta, **kwargs):
    """Azimuthal coordinate of an inbound photon at past null infinity

    This utility is a wrapper around `~scipy.integrate.quad` that integrates
    photon trajectories backward, tracing them to the far-field region. It
    uses geometrized units (``G = c = 1``) and is written in terms of
    ``r / M``, so all physical quantities are unitless.

    Parameters
    ----------
    r : `float`
        unitless radial coordinate

    delta : `float`
        angle (radians) between the ingoing radial direction and
        the instantaneous 4-velocity of a photon

    **kwargs : `dict`, optional
        additional keyword arguments to `~scipy.integrate.quad`

    Returns
    -------
    final : `float`
        azimuthal coordinate of the inbound photon, traced back to
        past null infinity

    See also
    --------
    scipy.integrate.quad
        the utility which numerically integrates at fixed ``delta``
    integrand
        integrand to find angular deflection along null geodesics
    """
    (delta, ) = numpy.unwrap([delta])
    b = impact_parameter(r, delta)
    # if outside the photon sphere,
    # integrate outgoing orbits
    if r > 3:
        dcrit = numpy.arccos(numpy.sqrt(2 / r))
        outgoing = (abs(delta) < dcrit)
        r0 = closest_approach(b)
        if outgoing and r0 == 0:
            warnings.warn(
                "Trapped orbit along angle = {}".format(delta),
                RuntimeWarning,
            )
            return numpy.nan
        elif outgoing:
            (final1, _) = quad(integrand, r0, r,
                               args=(b, ), **kwargs)
            (final2, _) = quad(integrand, r0, numpy.inf,
                               args=(b, ), **kwargs)
            return final1 + final2
    # else integrate ingoing orbits
    (final, _) = quad(integrand, r, numpy.inf,
                      args=(b, ), **kwargs)
    return final


def source_angle(r, ninterp=501, resol=ARCMIN, **kwargs):
    """Determine the source angle as a function of azimuthal angle

    This utility is a wrapper around :func:`pyschild.orbit.null.far_azimuth`
    that returns a callable function. At fixed ``r``, this function is the
    interpolated source angle as a function of local azimuthal angle away
    from the ingoing radial direction.

    Parameters
    ----------
    r : `float`
        unitless radial coordinate

    ninterp : `int`, optional
        number of points to sample (logarithmically) for interpolation,
        default: 501

    resol : `float`, optional
        angular resolution (radians) for determining black hole size,
        default: 0.000290888 (1 arcminute)

    **kwargs : `dict`, optional
        additional keyword arguments to `~scipy.integrate.quad`

    Returns
    -------
    angle : `callable`
        a continuous function of local azimuthal angle returning the
        source angle at fixed radial coordinate

    Notes
    -----
    The output callable function ``angle`` will raise a `ValueError` for
    photons that fall past the event horizon, i.e. for ingoing null rays
    with impact parameter ``b <= sqrt(27)``.

    See also
    --------
    far_azimuth
        the utility which numerically integrates along null geodesics
    scipy.interpolate.InterpolatedBivariateSpline
        the interpolation algorithm used here
    """
    psi = critical_angle(r) + resol
    delta = numpy.geomspace(psi, numpy.pi, num=ninterp)
    final = numpy.array([far_azimuth(r, d, **kwargs) for d in delta])
    # reject non-finite values and extend to 2*pi
    finite = (numpy.isfinite(final)).nonzero()
    domain = numpy.concatenate((
        delta[finite],
        (2 * numpy.pi - delta[finite][:-1])[::-1],
    ))
    range_ = numpy.concatenate((
        final[finite],
        (-final[finite][:-1])[::-1],
    ))
    # return interpolated function
    return InterpolatedUnivariateSpline(
        domain,
        range_,
        ext=2,
    )
