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

"""Core utilities for visualizing black hole orbits
"""

import numpy

from matplotlib import pyplot as plt

from gwpy.plot import Plot

from . import timelike
from .. import (
    utils,
    Signal,
)

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- utilities ----------------------------------------------------------------

def _potential_well_span(pot, level):
    """Convenience function to index the boundaries of a potential well
    """
    (allowed, ) = numpy.where(level - pot > 0)
    # get the highest-valued island of contiguous indices
    allowed = numpy.split(
        allowed,
        numpy.where(
            numpy.diff(allowed) != 1,
        )[0] + 1,
    )[-1]
    return (allowed[0], allowed[-1])


def potential(gamma, h, npoints=1001, xlim=None, ylim=None, fig=None,
              figsize=(12, 6), dpi=200, subplot=111, title=None):
    """Visualize the radial effective potential for timelike orbits

    Parameters
    ----------
    gamma : `array_like`
        generalized Lorentz factor (energy per unit rest mass)
        of the orbit(s) to visualize

    h : `float`
        unitless specific angular momentum

    npoints : `int`, optional
        number of points to sample uniformly, default: 1001

    xlim : `tuple` or `NoneType`, optional
        x-axis limits, default: choose based on the potential

    ylim : `tuple` or `NoneType`, optional
        y-axis limits, default: choose based on the potential

    fig : `int` or `NoneType`, optional
        figure number to attach, defaults to a new figure

    figsize : `tuple`, optional
        figure size in height x width (inches), default: ``(12, 6)``

    dpi : `int`, optional
        figure resolution per square inch, default: 200

    subplot : `int` with three digits, optional
        figure subplot indices to arrange this plot, default: ``111``

    title : `str` or `NoneType`, optional
        title for the figure, default: `None`

    Returns
    -------
    fig : `~matplotlib.pyplot.Figure`
        the populated figure object

    Raises
    ------
    ValueError
        if ``h`` does not support a circular orbit, i.e. if ``h**2 < 12``

    Examples
    --------
    To visualize a list of orbital Lorentz factors on a potential diagram:

    >>> from pyschild.orbit import plot
    >>> fig = plot.potential([0.01, 0.2, 0.8, 1.1])
    >>> fig.show()

    See also
    --------
    pyschild.orbit.timelike.radial_potential
        the (unitless) radial effective potential for timelike orbits
    """
    if not numpy.iterable(gamma):
        gamma = [gamma]
    energy = [(g - 1) / 2 for g in gamma]
    # determine axis limits
    href = max(h, timelike.HISCO)
    rplus = (href / 2) * (href + numpy.sqrt(href**2 - timelike.HISCO**2))
    pfid = -1 * timelike.radial_potential(rplus, h)
    xlim = xlim or (1e-10, 100 * rplus / 6)
    ylim = ylim or (-1.1 * pfid, max(0.75 * pfid, 1.1 * max(energy)))
    # get data
    r = numpy.linspace(*xlim, num=npoints)
    potential = timelike.radial_potential(r, h)
    # set up figure
    fig = plt.figure(num=fig, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(subplot, projection='rectilinear')
    # plot potential with energy
    ax.plot(xlim, (0, 0), color='#669999', linestyle='dashed')
    ax.plot(r, potential, zorder=100, linewidth=2, linestyle='dashdot',
            color='#333333', label='$h={:.4g}$'.format(h))
    for (g, eng) in zip(gamma, energy):
        (imin, imax) = _potential_well_span(potential, eng)
        ax.plot((r[imin], r[imax]), (eng, eng),
                label=r'$\gamma={:.4g}$'.format(g))
    # indicate the forbidden zone
    ax.fill_between(r, potential, ylim[0], hatch='/', alpha=0.4,
                    facecolor='#ee0000', edgecolor='k')
    ax.text(max(4 * rplus, 0.6 * (xlim[1] - xlim[0])),
            0.6 * ylim[0], 'FORBIDDEN', fontsize=16,
            va='top', bbox=dict(facecolor='white'))
    # return the figure
    ax.set_xlim(xlim)
    ax.set_xlabel('$c^2r/GM$')
    ax.set_ylim(ylim)
    ax.set_ylabel(r'$\Phi_h(r)/c^2$')
    ax.grid(True, which='major', axis='both', alpha=0.4)
    ax.legend(shadow=True)
    if isinstance(title, str):
        ax.set_title(title)
    return fig


def diagnostic(soln, duration, npoints=1000, figsize=(12, 12), dpi=200):
    """Display figures-of-merit for accuracy and precision of a simulation

    This utility produces a figure with five subplots, showing the timeseries
    records of ``r``, number of cycles, total 3-velocity, and the fractional
    errors in total energy and specific angular momentum, all with respect to
    (unitless) proper time.

    Parameters
    ----------
    soln : `~scipy.integrate.OdeSolution`
       interpolated solutions for ``r``, ``rdot``, and ``phi``,
       represented as a continuous function of proper time that
       returns an array of shape ``(3, n)``

    duration : `float`
        duration (in unitless proper time) of the simulated orbit

    npoints : `int`, optional
        number of points to sample uniformly, default: 1000

        .. note: This utility uses `~numpy.arange` with a step size of
                 ``duration / npoints``, in contrast with the other tools
                 in this module which use ``~numpy.linspace`` or
                 ``~pyschild.utils.power_sample``

    figsize : `tuple`, optional
        figure size in height x width (inches), default: ``(12, 12)``

    dpi : `int`, optional
        figure resolution per square inch, default: 200

    Returns
    -------
    fig : `~matplotlib.pyplot.Figure`
        the populated figure object

    Examples
    --------
    To get an extensive set of simulation figures-of-merit:

    >>> from pyschild.orbit import (timelike, plot)
    >>> y0 = timelike.initial_values(0.5)
    >>> (geodesic, duration) = timelike.simulate(y0, timelike.HISCO, tf=1000)
    >>> fig = plot.diagnostic(geodesic, duration)
    >>> fig.show()

    See also
    --------
    pyschild.orbit.timelike.simulate
        the utility which numerically simulates massive particle orbits
    pyschild.orbit.timelike.velocity
        compute the total 3-velocity given ``r``, ``rdot``, and ``phidot``
    """
    dt = duration / npoints
    times = numpy.arange(0, duration, dt)
    (r, rdot, phi) = soln(times)
    phidot = numpy.gradient(phi, dt)
    beta = timelike.velocity(r, rdot, phidot)
    # get constants of the motion
    heval = phidot * r**2
    geval = numpy.sqrt(rdot**2 + (1 + (heval / r)**2) * (1 - 2 / r))
    # get `Signal` objects for each variable
    r = Signal(r, dt=dt)
    ncyc = Signal((phi - phi[0]) / (2 * numpy.pi), dt=dt)
    beta = Signal(timelike.velocity(r, rdot, phidot), dt=dt)
    herror = Signal(numpy.abs((heval - heval[0]) / heval[0]) if heval[0]
                    else numpy.abs(heval - heval[0]),
                    dt=dt)
    gerror = Signal(numpy.abs((geval - geval[0]) / geval[0]) if geval[0]
                    else numpy.abs(geval - geval[0]),
                    dt=dt)
    # construct a figure with five panels
    fig = Plot(r, ncyc, beta, gerror, herror, figsize=figsize,
               separate=True, sharex=False, sharey=False)
    ax = fig.axes
    for i in range(4):
        ax[i].set_xlabel('')
        ax[i].set_xticklabels([])
        ax[i].grid(True, which='major', axis='both', alpha=0.4)
    ax[-1].set_xscale('seconds')
    ax[-1].set_xlabel(r'$c^3\tau/GM$')
    ax[0].plot((0, duration), (2, 2), color='#669999', linestyle='dashed')
    ax[0].set_ylabel('$c^2r/GM$')
    ax[0].set_ylim((0 if r.min().value < 2 else 0.9*r.min().value,
                   1.1*r.max().value))
    ax[1].set_ylabel('No. of cycles')
    ax[1].set_ylim((0, ncyc.max().value))
    ax[2].set_ylabel(r'$\beta$')
    ax[2].set_ylim((0, max(1, beta.max().value)))
    ax[3].set_ylabel(r'$\gamma_{{\mathrm{{err}}}}$ rel. '
                     'to {:.3g}'.format(geval[0]))
    gerr = gerror.value[~numpy.isnan(gerror.value)]
    ax[3].set_ylim((0.9 * gerr[1::].min(), min(gerr.max(), 1)))
    ax[3].set_yscale('log' if gerror[1::].value.min() else 'linear')
    ax[4].set_ylabel(r'$h_{{\mathrm{{err}}}}$ rel. '
                     'to {:.3g}'.format(heval[0]))
    herr = herror.value[~numpy.isnan(herror.value)]
    ax[4].set_ylim((0.9 * herr[1::].min(), min(herr.max(), 1)))
    ax[4].set_yscale('log' if herror[1::].value.min() else 'linear')
    fig.tight_layout()
    return fig


def track(solns, durations, npoints=1001, powersample=False,
          fig=None, figsize=(6, 6), dpi=200, title=None):
    """Draw the orbital tracks for a collection of simulations

    Parameters
    ----------
    solns : `list` of `~scipy.integrate.OdeSolution`
       interpolated solutions for ``r``, ``rdot``, and ``phi``,
       each is represented as a continuous function of proper
       time that returns an array of shape ``(3, n)``

    durations : `list` of `float`
        durations (in unitless proper time) of the simulated orbits

    npoints : `int`, optional
        number of points to sample, default: 1001

    powersample : `bool`, optional
        whether to use power-sampling to cluster points, recommended
        for capture orbits approaching the singularity, default:
        `False` (use uniform sampling)

    fig : `int` or `NoneType`, optional
        figure number to attach, defaults to a new figure

    figsize : `tuple`, optional
        figure size in height x width (inches), default: ``(12, 12)``

    dpi : `int`, optional
        figure resolution per square inch, default: 200

    title : `str` or `NoneType`, optional
        title for the figure, default: `None`

    Returns
    -------
    fig : `~matplotlib.pyplot.Figure`
        the populated figure object

    Examples
    --------
    To get an extensive set of simulation figures-of-merit:

    >>> from pyschild.orbit import (timelike, plot)
    >>> y0 = timelike.initial_values(0.5)
    >>> (geodesic, duration) = timelike.simulate(y0, timelike.HISCO, tf=1000)
    >>> fig = plot.track(geodesic, duration)
    >>> fig.show()

    See also
    --------
    pyschild.orbit.timelike.simulate
        the utility which numerically simulates massive particle orbits
    pyschild.utils.power_sample
        the utility which handles power-sampling
    """
    rmax = 0
    space = (utils.power_sample if powersample
             else numpy.linspace)
    # cast input objects to list
    if not numpy.iterable(solns):
        solns = [solns]
    if not numpy.iterable(durations):
        durations = [durations]
    # set up a radial figure
    fig = plt.figure(num=fig, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='polar')
    for (duration, soln) in zip(durations, solns):
        times = space(0, duration, num=npoints)
        (r, _, phi) = soln(times)
        rmax = max(rmax, r.max())
        # draw the orbital curve
        ax.plot(phi, r)
    # return the figure
    ax.set_rlim((0, 1.05 * rmax))
    ax.grid(True, alpha=0.4)
    if isinstance(title, str):
        ax.set_xticks([0, numpy.pi/4, 3*numpy.pi/4, numpy.pi,
                       5*numpy.pi/4, 3*numpy.pi/2, 7*numpy.pi/4])
        ax.set_title(title)
    return fig
