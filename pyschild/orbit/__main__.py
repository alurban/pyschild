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

"""Command-line interface for simulations of black hole orbits
"""

import numpy
import os
import sys
import time

from astropy.constants import (G, c)
from astropy.units import Unit

from gwpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
)

from matplotlib import use
# if run in an environment with no display,
# use the 'Agg' backend to render plots
if len(os.getenv('DISPLAY', '')) == 0:
    use('Agg')
from matplotlib import pyplot as plt  # noqa: E402

from .. import (  # noqa: E402
    cli,
    utils,
)
from . import (  # noqa: E402
    timelike,
    plot,
)

__author__ = "Alex Urban <alexander.urban@ligo.org>"

# logging interface
LOGGER = cli.logger(name="pyschild.orbit")

# base unit
SECOND = Unit("sec")


# -- command-line interface ---------------------------------------------------

def populate_arguments(parser):
    """Flesh out command-line arguments
    """
    parser.add_argument('-o', '--output-dir', default=os.getcwd(),
                        help="output directory for data products, "
                             "default: %(default)s")
    parser.add_argument('-l', '--lorentz-factor', type=float, default=1,
                        help="energy per unit rest mass along this orbit, "
                             "must be no less than 1 - 2 / r0 where r0 is "
                             "the initial radial coordinate, "
                             "default: %(default)s")
    parser.add_argument('-a', '--angular-momentum', type=float, default=0,
                        help="unitless specific angular momentum along this "
                             "orbit, default: %(default)s")
    parser.add_argument('-r', '--initial-radius', type=float, default=None,
                        help="initial radial coordinate along this orbit, "
                             "defaults to the largest possible apsis")
    parser.add_argument('-d', '--duration', type=float, default=60,
                        help="duration (seconds) of the final simulation, "
                             "will determine the mass of the black hole, "
                             "default: %(default)s")
    parser.add_argument('-t', '--step-size', type=float, default=3e-2,
                        help="step size (seconds) over which to sample the "
                             "simulation output, default: %(default)s")
    return parser


def parse_command_line(args=None):
    """Parse command-line arguments
    """
    parser = cli.create_parser(description=__doc__)
    parser = populate_arguments(parser)
    return parser.parse_args(args=args)


# -- run ----------------------------------------------------------------------

def make_output_directory(outdir):
    """Create the output directory, if it does not already exist
    """
    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        action = LOGGER.debug
        verb = "Created"
        clause = ""
    else:
        action = LOGGER.warning
        verb = "Found"
        clause = ", any data products will be overwritten"
    action("{0} output directory at {1}{2}".format(verb, outdir, clause))
    return outdir


def get_signals(r, rdot, phi, dt, mass):
    """Create a `~gwpy.timeseries.TimeSeriesDict` based on orbital trajectory
    """
    rconv = G * mass / c**2
    tconv = G * mass / c**3
    dt *= tconv
    r = TimeSeries(rconv * r, dt=dt.value,
                   name="radial coordinate along geodesic")
    rdot = TimeSeries(c * rdot, dt=dt.value,
                      name="radial component of 4-velocity")
    phi = TimeSeries(phi, dt=dt.value,
                     name="azimuthal coordinate along geodesic")
    return TimeSeriesDict({'r': r, 'rdot': rdot, 'phi': phi})


def main(args=None):
    """Run pyschild-orbit on the command-line

    This utility returns the exit code at runtime,
    which can be passed to :func:`sys.exit`.
    """
    # parse command-line options and arguments
    args = parse_command_line(args=args)
    gamma = args.lorentz_factor
    h = args.angular_momentum
    r0 = args.initial_radius

    # capture start time for verbose output
    LOGGER.info("Launching pyschild.orbit")
    outdir = make_output_directory(args.output_dir)
    start = time.time()

    # perform simulation
    LOGGER.debug("Integrating orbit with gamma = {:.2f}, "
                 "h = {:.2f}".format(gamma, h))
    initial = timelike.initial_values(gamma, h, r0=r0, ingoing=True)
    (soln, duration) = timelike.simulate(initial, h)

    # match requested duration to source mass
    mass = ((c**3 * args.duration * SECOND) /
            (G * duration)).to("Msun")
    msymb = utils.format_scientific(mass.value)
    LOGGER.info("Source mass: {:.2e}".format(mass))

    # discretize the numerical solution
    dt = ((c**3 * args.step_size * SECOND) / (G * mass)).value
    times = numpy.arange(0, duration, dt)
    (r, rdot, phi) = soln(times)

    # capture the time taken to simulate
    tsim = time.time() - start
    LOGGER.debug("Time taken to simulate: {:.2e} seconds".format(tsim))

    # plot potential well diagram
    well = plot.potential(
        gamma, h, title="Radial potential well, $M = {}$ "
                        r"$M_\odot$".format(msymb))
    well.savefig(os.path.join(outdir, 'potential-well.png'), dpi=200)
    plt.close()

    # plot simulation figures of merit
    fom = plot.diagnostic(soln, duration)
    fom.savefig(os.path.join(outdir, 'figures-of-merit.png'), dpi=200)
    plt.close()

    # plot orbital track
    track = plot.track(soln, duration, powersample=(r[-1] < 2),
                       title="Orbital track, $M = {}$ "
                             r"$M_\odot$".format(msymb))
    track.savefig(os.path.join(outdir, 'orbital-track.png'), dpi=200)
    plt.close()

    # capture the time taken to render plots
    tren = time.time() - (start + tsim)
    LOGGER.debug("Time taken to render plots: {:.2e} seconds".format(tren))

    # write data products to output directory
    products = get_signals(r, rdot, phi, args.step_size, mass)
    products.write(os.path.join(outdir, 'orbital-trajectory.h5'),
                   format='hdf5', overwrite=True)


if __name__ == "__main__":  # pragma: no-cover
    sys.exit(main())
