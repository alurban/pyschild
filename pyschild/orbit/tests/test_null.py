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
import pytest

from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import (argrelmin, argrelmax)

from ... import utils
from .. import null

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- test utilities -----------------------------------------------------------

def test_radial_potential():
    """Test :func:`pyschild.orbit.null.radial_potential`
    """
    r = numpy.arange(1, 100, 0.1)

    # test with no angular momentum
    p1 = null.radial_potential(r, 0)
    assert utils.zero_crossings(numpy.diff(p1)).size == 0

    # test with angular momentum that supports stable circular orbits
    p2 = null.radial_potential(r, 4)
    (pmin, ) = argrelmin(p2)
    (pmax, ) = argrelmax(p2)
    assert pmin.size == 0
    assert pmax.size == 1
    assert_allclose(r[pmax], 3)  # photon sphere


def test_impact_parameter():
    """Test :func:`pyschild.orbit.null.impact_parameter`
    """
    r = 30
    psi = null.critical_angle(r)
    b = null.impact_parameter(r, psi)
    assert_allclose(b, numpy.sqrt(27))


def test_closest_approach():
    """Test :func:`pyschild.orbit.null.closest_approach`
    """
    # test with small impact parameter
    assert null.closest_approach(0) == 0

    # test with marginal impact parameter
    assert_allclose(null.closest_approach(numpy.sqrt(27)), 3)


def test_critical_angle():
    """Test :func:`pyschild.orbit.null.critical_angle
    """
    # test at singularity
    assert null.critical_angle(0) == numpy.pi / 2

    # test at horizon crossing
    assert_allclose(null.critical_angle(2), numpy.arccos(23 / 31))


def test_integrand():
    """Test :func:`pyschild.orbit.null.integrand`
    """
    b = 4
    r = numpy.linspace(1, 50, 101)
    denom = r * numpy.sqrt((r/b)**2 - (1 - 2/r))
    assert_allclose(
        null.integrand(r, b),
        -1 / denom,
    )


def test_far_azimuth():
    """Test :func:`pyschild.orbit.null.far_azimuth`
    """
    # test an escaped photon
    escaped = null.far_azimuth(6, numpy.pi)
    assert_allclose(escaped, 0, atol=1.5e-16)

    # test an absorbed photon
    with pytest.warns(RuntimeWarning) as record:
        absorbed = null.far_azimuth(6, 0)
    assert record[0].message.args[0] == (
        "Trapped orbit along angle = 0.0")
    assert numpy.isnan(absorbed)

    # test from behind the horizon
    incoming = null.far_azimuth(1, numpy.pi)
    assert_allclose(incoming, 0, atol=1e-16)


def test_source_angle():
    """Test :func:`pyschild.orbit.null.source_angle`
    """
    angle = null.source_angle(20)
    assert isinstance(angle, InterpolatedUnivariateSpline)
    assert_allclose(angle(numpy.pi), 0, atol=1.5e-16)
    with pytest.raises(ValueError):
        angle(0)
