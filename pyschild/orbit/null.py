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

from scipy.integrate import solve_ivp
from scipy.linalg import norm

__author__ = "Alex Urban <alexander.urban@ligo.org>"
__credit__ = "Riccardo Antonelli <http://rantonels.github.io>"


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


def rhs(_t, y):
    """Right-hand side of the parameterized Binet equation for null geodesics

    This is a convenience function to enable numerical solutions of
    Schwarzschild orbits given a value for ``r`` and ``rdot`` at the
    same point(s). It uses geometrized units (``G = c = 1``) and is
    written in terms of ``r / M``, so all physical quantities are unitless.

    Parameters
    ----------
    _t : `float`
        timestamp of the solution point (not used in practice, but needed
        in the function signature by `~scipy.integrate.solve_ivp`)

    y : `array-like`
        collection of (unitless) solution values for
        ``[x, y, z, xdot, ydot, zdot]``

    Returns
    -------
    dydt : `~numpy.ndarray`
        an array of derivatives for ``[x, y, z, xdot, ydot, zdot]``
        evaluated at the given point

    Notes
    -----
    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``h`` and all terms of the potential are unitless.

    See also
    --------
    scipy.integrate.solve_ivp
        the utility which calls this function to numerically solve
        null geodesic equations of motion
    """
    (x, xdot) = numpy.split(y, 2)
    h = norm(numpy.cross(x, xdot))
    r = norm(x)
    # accelerations
    xddot = numpy.array([
        # we are integrating *backwards*, so this is off
        # by a minus sign from the Binet equation
        3 * h**2 * elem / (2 * r**5) for elem in (x / r)
    ])
    return numpy.concatenate((xdot, xddot))


def simulate(y0, tf=1000, **kwargs):
    """Numerically integrate parameterized Binet equations for null geodesics

    This utility is a wrapper around `~scipy.integrate.solve_ivp` that
    integrates backward with a custom termination event for photons that
    reach the far-field region. It uses geometrized units (``G = c = 1``)
    and is written in terms of ``r / M``, so all physical quantities are
    unitless.

    Parameters
    ----------
    y0 : `array-like`
        array of initial values in the form ``[x, y, z, xdot, ydot, zdot]``

    tf : `float`, optional
        unitless termination time of the simulation, default: 1000

    **kwargs : `dict`, optional
        additional keyword arguments to `~scipy.integrate.solve_ivp`

    Returns
    -------
    soln : `~scipy.integrate.OdeSolution`
       interpolated solutions for ``x``, ``y``, ``z``, ``xdot``, ``ydot``,
       and ``zdot``, represented as a continuous function of proper time that
       returns an array of shape ``(6, n)``

    duration : `float`
        last timestamp of the null ray, either ``tf`` (for rays that do not
        escape in time) or the approximate time at which the ray "reaches"
        the far-field region (measured by the Binet force shrinking to
        ``1e-10``)

        ..note: Because there is no well-defined notion of proper time for
                null geodesics, the curve parameter ``t`` **does not**
                measure time in any meaningful sense.

    Notes
    -----
    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so all physical quantities are unitless. However, any
    solution can easily be re-scaled to physical units, see below.

    Because the simulation integrates six dependent variables as a
    function of proper time, ``soln`` returns an array of shape ``(6, )``
    for every individual timestamp, and an array of shape ``(6, n)`` for
    a timestamp array of length ``n``.

    Users may still pass events using the ``events`` keyword argument, which
    should be a list of functions that each accept both and only ``(t, r)``
    as arguments (in that order).

    Examples
    --------
    We can trace a photon trajectory backwards from the instant it crosses
    the photon sphere:

    >>> import numpy
    >>> from pyschild.orbit import null
    >>> initial = numpy.array([3, 0, 0, 1, 0, 0])
    >>> (soln, duration) = null.simulate(initial)

    Then, create an array of curve parameters to sample the interpolated
    result:

    >>> param = numpy.arange(duration)
    >>> (x, y, z, xdot, ydot, zdot) = soln(param)

    Since ``soln(param)`` is a 6 x n array, we can directly unpack it to
    get ``x``, ``y``, ``z``, and their derivatives as separate 1-D arrays
    each the same length as ``param``.

    To convert these into physical units, first let ``m`` be a mass in solar
    masses, then use base unit objects, e.g.:

    >>> from astropy import units
    >>> from pyschild.orbit import (RSOL, TSOL)
    >>> m = 1.4  # typical mass of a neutron star
    >>> x *= m * RSOL
    >>> xdot *= (RSOL / TSOL)

    See also
    --------
    scipy.integrate.solve_ivp
        utility which numerically integrates to solve an initial value
        problem
    rhs
        right-hand side of the parameterized Binet equation
    """
    def far_field(_t, y):
        """Terminal event for a photon reaching the far-field

        The ``_t`` positional argument is a dummy variable required by the
        numerical integrator. For details see `~scipy.integrate.solve_ivp`
        """
        (x, xdot) = numpy.split(y, 2)
        h = norm(numpy.cross(x, xdot))
        r = norm(x)
        return (3 * h**2 / (2 * r**5) - 1e-10)
    far_field.terminal = True
    far_field.direction = -1
    # append far_field to the list of events
    events = kwargs.pop('events', [])
    events.append(far_field)
    # perform numerical integration
    kwargs.setdefault('atol', 1e-6 * tf / 100)
    soln = solve_ivp(rhs, (0, tf), y0, dense_output=True,
                     events=events, **kwargs)
    # return interpolated solution
    return (soln.sol, soln.t[-1])
