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

"""Tests for :mod:`pyschild.orbit.null`
"""

import numpy

from numpy.testing import assert_allclose
from scipy.signal import (argrelmin, argrelmax)

from ... import utils
from .. import null

__author__ = "Alex Urban <alexander.urban@ligo.org>"

ANGMOM = 4
R = numpy.arange(1, 100, 0.1)


# -- test utilities -----------------------------------------------------------

def test_radial_potential():
    """Test :func:`pyschild.orbit.null.radial_potential`
    """
    # test with no angular momentum
    p1 = null.radial_potential(R, 0)
    assert utils.zero_crossings(numpy.diff(p1)).size == 0

    # test with angular momentum that supports stable circular orbits
    p2 = null.radial_potential(R, ANGMOM)
    (pmin, ) = argrelmin(p2)
    (pmax, ) = argrelmax(p2)
    assert pmin.size == 0
    assert pmax.size == 1
    assert_allclose(R[pmax], 3)  # photon sphere


def test_rhs():
    """Test :func:`pyschild.orbit.null.rhs`
    """
    # radial infall on the photon sphere
    y0 = numpy.array([0, 0, 3, 0, 0, 1])
    rhs = null.rhs(0, y0)
    assert_allclose(rhs, [0, 0, 1, 0, 0, 0])


def test_simulate():
    """Test :func:`pyschild.orbit.null.simulate`
    """
    # test from photon sphere
    y0 = numpy.array([0, 0, 3, 0, 1, 0])
    (psph, duration) = null.simulate(y0)
    times = numpy.arange(duration)
    (x, y, z, xdot, ydot, zdot) = psph(times)
    r = numpy.sqrt(x**2 + y**2 + z**2)
    v = numpy.sqrt(xdot**2 + ydot**2 + zdot**2)
    assert (r >= 3).all()
    assert numpy.diff(r).all()
    assert_allclose(v, 1, rtol=0.05)
    assert duration < 1000
