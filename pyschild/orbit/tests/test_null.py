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
    psi = numpy.arcsin(numpy.sqrt(27) / r)
    b = null.impact_parameter(r, psi)
    assert_allclose(b, numpy.sqrt(27), rtol=1.5e-3)


def test_closest_approach():
    """Test :func:`pyschild.orbit.null.closest_approach`
    """
    # test with small impact parameter
    assert null.closest_approach(0) == 0

    # test with marginal impact parameter
    assert_allclose(null.closest_approach(3 * numpy.sqrt(3)), 3)


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


def test_phi_inf():
    """Test :func:`pyschild.orbit.null.phi_inf`
    """
    # test an escaped photon
    escaped = null.phi_inf(6, numpy.pi)
    assert_allclose(escaped, 0, atol=1.5e-16)

    # test an absorbed photon
    with pytest.warns(RuntimeWarning) as record:
        absorbed = null.phi_inf(6, 0)
    assert record[0].message.args[0] == (
        "Orbit along delta = 0.0 is absorbed")
    assert numpy.isnan(absorbed)


def test_source_angle():
    """Test :func:`pyschild.orbit.null.source_angle`
    """
    angle = null.source_angle(20)
    assert isinstance(angle, InterpolatedUnivariateSpline)
    assert_allclose(angle(numpy.pi), 0, atol=1.5e-16)
    with pytest.raises(ValueError):
        angle(0)
