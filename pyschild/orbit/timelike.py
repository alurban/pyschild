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

def _stationary_aberration_angle(delta, r):
    """Convenience tool to get the aberration angle for stationary observers

    See :func:`aberration_angle` for the general utility.
    """
    if numpy.any(r < 2):
        raise ValueError("No stationary observers behind the event horizon")
    from .null import impact_parameter
    b = impact_parameter(r, numpy.pi - delta)
    dcrit = numpy.pi - numpy.arccos(numpy.sqrt(2 / r))
    sgn = numpy.piecewise(delta, (delta >= dcrit, delta < dcrit), (-1, 1))
    return numpy.arccos(sgn * numpy.sqrt(1 - (1 - 2 / r) * (b / r)**2))


def radial_potential(r, h):
    """Effective radial potential for timelike Schwarzschild orbits

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
    >>> from pyschild.orbit.timelike import radial_potential
    >>> r = numpy.arange(20)
    >>> potential = radial_potential(r, 12)
    >>> r *= G * M_sun / c**2
    >>> potential *= c**2

    See also
    --------
    radial_force
        a utility to compute the effective radial force for timelike geodesics
    pyschild.orbit.null.radial_potential
        effective radial potential for null geodesics
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
        unitless specific orbital angular momentum

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


def velocity(r, rdot, phidot):
    """Magnitude of the local 3-velocity of a massive particle

    This observable is measured relative to a stationary observer, which can
    be misleading inside the event horizon, where stationary observers do not
    exist yet the formula for 3-velocity can be smoothly extended. In this
    region the 3-velocity appears superluminal only because ``r`` becomes a
    timelike coordinate.

    Parameters
    ----------
    r : `array_like`
        unitless radial coordinate along a geodesic

    rdot : `array_like`
        unitless derivative of ``r`` with respect to proper time
        along a geodesic

    phidot : `array_like`
        unitless derivative of azimuthal coordinate with respect
        to proper time along a geodesic

    Returns
    -------
    beta : `float` or `~numpy.ndarray`
        magnitude of the local unitless 3-velocity

    References
    ----------
    .. [1] Wikipedia, "Schwarzschild geodesics,"
           https://en.wikipedia.org/wiki/Schwarzschild_geodesics
    """
    h = r**2 * phidot
    gamma2 = 1 + rdot**2 + 2 * radial_potential(r, h)
    return numpy.sqrt(1 - (1 - 2 / r) / gamma2)


def aberration_angle(delta, r, rdot, phidot):
    """Determine the aberration angle for relativistic orbital motion

    This observable measures the apparent angle of incidence for incoming
    photons relative to an observer in motion along a timelike geodesic.
    The original angle, ``delta``, describes what would be measured at the
    same altitude by a radially plunging observer who started from rest at
    infinity.

    Parameters
    ----------
    delta : `array_like`
        viewing angle relative to a radially plunging observer

    r : `array_like`
        unitless radial coordinate along a geodesic

    rdot : `array_like`
        unitless derivative of ``r`` with respect to proper time
        along a geodesic

    phidot : `array_like`
        unitless derivative of azimuthal coordinate with respect
        to proper time along a geodesic

    Returns
    -------
    psi : `float` or `~numpy.ndarray`
        angle of incidence measured by the observer along this geodesic

    References
    ----------
    .. [1] D. Lebedev and K. Lake, arXiv:1609.05183 (2016)

    .. [2] H. Arakida, arXiv:1808.03418

    See also
    --------
    pyschild.sky.SkyMap.aberrate
        method to impose relativistic aberration on an observer's entire sky
    """
    # stationary observers
    if numpy.array_equal(rdot, 0) and numpy.array_equal(phidot, 0):
        return _stationary_aberration_angle(delta, r)
    # geometric quantities
    gamma = numpy.sqrt(
        1 + rdot**2 +  # generalized Lorentz factor
        2 * radial_potential(r, r**2 * phidot))
    sqrtr = numpy.sqrt(r / 2)
    sind = numpy.sin(delta)
    cosd = numpy.cos(delta)
    # local velocities
    # transverse velocity is cast to complex to handle r < 2
    vperp = numpy.sqrt(r * (r - 2) + 0j) * phidot / gamma
    vpar = rdot / gamma
    vsq = vpar**2 + (vperp**2).real
    # vector products
    # factor by `gtt * (1 - vsq)` for cleanliness
    # and cast a product of imaginary numbers to float
    wdotk = (r - 2) * (1 - vsq) * (1 - cosd)
    udotw = 2 * (1 - sqrtr) * (1 - vpar)
    udotk = (sqrtr + vpar - (1 + sqrtr * vpar) * cosd -
             (numpy.sqrt(r / 2 - 1 + 0j) * vperp).real * sind)
    # compute aberration angle
    ok = numpy.asarray(r != 2)
    if numpy.all(ok):
        return numpy.arccos(1 + wdotk / (udotk * udotw))
    # evaluate the limit as r --> 2
    psi = (numpy.ones_like(wdotk) *
           aberration_angle(delta, 2 + 1e-15, rdot, phidot))
    if numpy.iterable(ok):
        psi[ok] = numpy.arccos(1 + wdotk[ok] / (udotk[ok] * udotw[ok]))
    return psi


def initial_values(gamma, h, r0=None, phi0=0, ingoing=True):
    """Determine initial conditions for timelike orbits

    This utility calculates starting values for ``r``, ``rdot``, and ``phi``
    given an orbital eccentricity and orbital angular momentum. It uses
    geometrized units (``G = c = 1``) and is written in terms of ``r / M``,
    so ``h`` and all other physical quantities are unitless.

    Parameters
    ----------
    gamma : `float`
        generalized Lorentz factor equal to the total energy per rest mass,
        values less than 1 are bound orbits while those greater than 1 are
        unbound

    ..note: For a given value of ``r``, the factor ``gamma`` must
            be larger than ``(1 + h**2 / r**2) * (1 - 2 / r)``
            and no smaller than zero

    h : `float`
        unitless specific orbital angular momentum

    r0 : `float`, optional
        initial radial coordinate, defaults to either the largest possible
        apsis (if ``gamma < 1``) or the largest apsis of a nearly-marginal
        orbit (if ``gamma >= 1``)

    phi0 : `float`, optional
        orbital phase offset in radians, default: 0

    ingoing : `bool`, optional
        whether the orbit is initially falling toward the black hole (`True`)
        or moving away from it (`False`), default: `True`

    Returns
    -------
    y0 : `~numpy.ndarray`
        array of initial values for ``r``, ``rdot``, and ``phi``

    Raises
    ------
    ValueError
        if the Lorentz factor is too small to be physical
    """
    sgn = -1 if ingoing else 1
    roots = numpy.roots([
        1 - min(gamma, 0.99)**2,
        -2, h**2, -2 * h**2,
    ])
    # get initial values
    r0 = r0 or roots[numpy.isreal(roots)].max().real
    gcrit = numpy.sqrt((1 + (h / r0)**2) * (1 - 2 / r0))
    if gamma < gcrit:
        raise ValueError("Lorentz factor is too small "
                         "for the requested radius")
    rdot0 = sgn * numpy.sqrt(gamma**2 - gcrit**2)
    return numpy.array([r0, rdot0, phi0])


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
        unitless specific orbital angular momentum

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
        unitless specific orbital angular momentum

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

    Users may still pass events using the ``events`` keyword argument, which
    should be a list of functions that each accept all and only ``(t, r, h)``
    as arguments (in that order).

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
    # append singularity to the list of events
    events = kwargs.pop('events', [])
    events.append(singularity)
    # perform numerical integration
    # by default, integrate for 10 circular
    # orbits according to Kepler's third law
    tf = tf or 20 * numpy.pi * y0[0] ** (3/2)
    kwargs.setdefault('atol', 1e-6 * tf / 100)
    soln = solve_ivp(rhs, (0, tf), y0, args=(h, ),
                     dense_output=True, events=events,
                     **kwargs)
    if verbose:  # print verbose output
        print(VERBOSE.format(code=int(not soln.success), nfev=soln.nfev,
                             status=soln.status, message=soln.message))
    if not soln.success:  # warn on failure
        warnings.warn(soln.message, RuntimeWarning)
    # return interpolated solution
    return (soln.sol, soln.t[-1])
