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

"""Core utilities for simulating timelike geodesics
"""

import numpy
import warnings

from scipy.integrate import solve_ivp

__author__ = "Alex Urban <alexander.urban@ligo.org>"

# ISCO orbital angular momentum
HISCO = numpy.sqrt(12)

# verbose output
VERBOSE = """
==================
Simulation summary
==================

This simulation finished with status {code}
No. of RHS evaluations: {nfev}
Termination reason: {status}
Output message: {message}
"""


# -- utilities ----------------------------------------------------------------

def radial_potential(r, h):
    """Effective radial potential for timelike Schwarzschild orbits

    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``h`` and all terms of the potential are unitless.

    Parameters
    ----------
    r : `float` or `array-like`
        unitless radial coordinate

    h : `float`
        unitless orbital angular momentum

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
    >>> from pyschild.orbit.timelike import radial_potential
    >>> r = numpy.arange(20)
    >>> potential = radial_potential(r, 12)
    >>> r *= G * M_sun / c**2
    >>> potential *= c**2

    See also
    --------
    radial_force
        a utility to compute the effective radial force for timelike geodesics
    """
    return -1 / r + (h/r)**2 * (1 - 2 / r) / 2


def radial_force(r, h):
    """Effective radial force for timelike Schwarzschild orbits

    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``h`` and all terms of the force are unitless.

    Parameters
    ----------
    r : `float` or `array-like`
        unitless radial coordinate

    h : `float`
        unitless orbital angular momentum

    Returns
    -------
    force : `float` or `~numpy.ndarray`
        effective radial force at the points represented by ``r``

    Examples
    --------
    To put this quantity in physical units, it is recommended to use ``G``,
    ``M_sun``, and ``c`` from `~astropy.constants` which explicitly carry
    units, e.g.

    >>> import numpy
    >>> from astropy.constants import (G, M_sun, c)
    >>> from pyschild.orbit.timelike import radial_force
    >>> r = numpy.arange(20)
    >>> force = radial_force(r, 12)
    >>> r *= G * M_sun / c**2
    >>> force *= c**4 / (G * M_sun)

    See also
    --------
    radial_potential
        a utility to compute the effective radial potential for timelike
        geodesics
    """
    return -1 / r**2 + h**2 / r**3 - 3 * h**2 / r**4


def initial_values(ecc, h=HISCO, phi0=numpy.pi/2):
    """Determine initial conditions for timelike orbits

    This utility calculates starting values for ``r``, ``rdot``, and ``phi``
    given an orbital eccentricity and orbital angular momentum. It uses
    geometrized units (``G = c = 1``) and is written in terms of ``r / M``,
    so ``h`` and all other physical quantities are unitless.

    Parameters
    ----------
    ecc : `float`
        orbital eccentricity, must be greater than or equal to zero,
        values less than 1 are bound orbits while those greater than 1
        are unbound

    h : `float`, optional
        unitless orbital angular momentum, must be no smaller than
        `numpy.sqrt(12)` so the potential can support a circular orbit

    phi0 : `float`, optional
        orbital phase offset, defaults to `~numpy.pi/2`

    Returns
    -------
    y0 : `~numpy.ndarray`
        array of initial values for ``r``, ``rdot``, and ``phi``

    Raises
    ------
    ValueError
        if ``h`` does not support a circular orbit, i.e. if ``h**2 < 12``

    Notes
    -----
    This utility adopts a convention in which every orbit (bound or unbound)
    starts outgoing at the radius of a stable local minimum in the potential
    with given ``h``. In Newtonian physics, this is analogous to starting
    every orbit at the semi-latus rectum with radial velocity determined
    by ``ecc`` and ``h``.

    The default for orbital phase is chosen such that the analogous Newtonian
    orbit has periastron (closest approach) at ``phi = 0``.
    """
    if h < HISCO:
        raise ValueError("Orbital angular momentum must exceed ISCO "
                         "(where h_ISCO**2 = 12)")
    # express in terms of circular orbits
    rcirc = (h / 2) * (h + numpy.sqrt(h**2 - HISCO**2))
    ecirc = radial_potential(rcirc, h)
    energy = ecirc * (1 - ecc**2)
    drdt = numpy.sqrt(
        2 * (energy -
             radial_potential(rcirc, h)))
    return numpy.array([rcirc, drdt, phi0])


def rhs(_t, y, h):
    """Right-hand sides of timelike geodesic equations of motion

    This is a convenience function to enable numerical solutions of
    Schwarzschild orbits given a value for r at the same point(s). It
    uses geometrized units (``G = c = 1``) and is written in terms of
    ``r / M``, so all physical quantities are unitless.

    Parameters
    ----------
    _t : `float`
        timestamp of the solution point (not used in practice, but needed
        in the function signature by `~scipy.integrate.solve_ivp`)

    y : `array-like`
        collection of (unitless) solution values for
        ``r``, ``v_r``, and ``phi``

    h : `float`
        angular momentum per unit mass (unitless)

    Returns
    -------
    dydt : `~numpy.ndarray`
        an array of derivatives for ``r``, ``v_r``, and ``phi``
        evaluated at the given point

    Notes
    -----
    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so ``h`` and all terms of the potential are unitless.

    See also
    --------
    scipy.integrate.solve_ivp
        the utility which calls this function to numerically solve
        timelike geodesic equations of motion
    """
    (r, vr, _) = y
    # velocities
    rdot = vr
    vrdot = radial_force(r, h)
    phidot = h / r**2
    return numpy.array([rdot, vrdot, phidot])


def simulate(y0, h, tf=None, verbose=False, **kwargs):
    """Numerically integrate timelike geodesic equations of motion

    This utility is a wrapper around `~scipy.integrate.solve_ivp` with a
    custom termination event for particles that strike ``r = 0``. It uses
    geometrized units (``G = c = 1``) and is written in terms of ``r / M``,
    so all physical quantities are unitless.

    Parameters
    ----------
    y0 : `array-like`
        array of initial values for ``r``, ``rdot``, and ``phi``

    h : `float`
        unitless orbital angular momentum parameter

    tf : `float`, optional
        unitless termination time of the simulation, defaults to
        ``20 * numpy.pi * y0[0] ** (3/2)`` (i.e., 10 cycles according
        to Kepler's third law)

    verbose : `bool`, optional
        if `True`, prints verbose output about the final status of the
        simulation, default: `False`

    **kwargs : `dict`, optional
        additional keyword arguments to `~scipy.integrate.solve_ivp`

    Returns
    -------
    soln : `~scipy.integrate.OdeSolution`
       interpolated solutions for ``r``, ``rdot``, and ``phi``,
       represented as a continuous function of proper time that
       returns an array of shape ``(3, n)``

    duration : `float`
        last timestamp of the simulation, either ``tf`` (for non-capture
        orbits) or the time at which this orbit reaches the singularity

    Warnings
    --------
    RuntimeWarning
        if the integration does not converge or fails for any reason
        (will report the reason to `stderr`)

    Notes
    -----
    This uses geometrized units (``G = c = 1``) and is written in terms
    of ``r / M``, so all physical quantities are unitless. However, any
    solution can easily be re-scaled to physical units, see below.

    Because the simulation integrates three dependent variables as a
    function of proper time, ``soln`` returns an array of shape ``(3, )``
    for every individual timestamp, and an array of shape ``(3, n)`` for
    a timestamp array of length ``n``.

    Examples
    --------
    The object returned is a callable function that interpolates numerically
    integrated variables. For example, we can simulate a circular orbit:

    >>> from pyschild.orbit import timelike
    >>> initial = timelike.initial_values(0, h=50)
    >>> (soln, duration) = timelike.simulate(initial, h=50)

    Then, create an array of timestamps to sample the interpolated result:

    >>> import numpy
    >>> times = numpy.arange(duration)
    >>> (r, rdot, phi) = soln(times)

    Since ``soln(times)`` is a 3 x n array, we can directly unpack it to
    get ``r``, ``rdot``, and ``phi`` as separate 1-D arrays each the same
    length as ``times``.

    To convert these into physical units, first let ``m`` be a mass in solar
    masses, then use base unit objects:

    >>> from astropy import units
    >>> from pyschild.orbit import (RSOL, TSOL)
    >>> m = 1.4  # typical mass of a neutron star
    >>> times *= m * TSOL
    >>> r *= m * RSOL
    >>> rdot *= (RSOL / TSOL)
    >>> phi *= units.Unit("rad")

    See also
    --------
    scipy.integrate.solve_ivp
        utility which numerically integrates to solve an initial value
        problem
    initial_values
        convenience function to compute initial values for an orbit of given
        eccentricity and orbital angular momentum
    rhs
        right-hand sides of the geodesic equations of motion
    """
    def singularity(_t, y, _h):
        """Terminal event for a particle striking ``r = 0``

        The ``_t`` and ``_h`` positional arguments are dummy variables
        required by the numerical integrator. For details please see
        `~scipy.integrate.solve_ivp`
        """
        return y[0]
    singularity.terminal = True
    singularity.direction = -1
    # perform numerical integration
    # by default, integrate for 10 circular
    # orbits according to Kepler's third law
    tf = tf or 20 * numpy.pi * y0[0] ** (3/2)
    kwargs.setdefault('atol', 1e-6 * tf / 100)
    soln = solve_ivp(rhs, (0, tf), y0, args=(h, ),
                     dense_output=True, events=singularity,
                     **kwargs)
    if verbose:  # print verbose output
        print(VERBOSE.format(code=int(not soln.success), nfev=soln.nfev,
                             status=soln.status, message=soln.message))
    if not soln.success:  # warn on failure
        warnings.warn(soln.message, RuntimeWarning)
    # return interpolated solution
    return (soln.sol, soln.t[-1])
