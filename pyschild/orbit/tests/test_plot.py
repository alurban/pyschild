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

"""Tests for :mod:`pyschild.orbit.plot`

..note: These tests do not check that the output is correct, only that
        plotting utilities run to completion.
"""

import pytest

from matplotlib.pyplot import Figure

from .. import (timelike, plot)

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- test utilities -----------------------------------------------------------

def test_potential():
    """Test :func:`pyschild.orbit.plot.potential`
    """
    fig = plot.potential(0.5, title="Test potential")
    assert isinstance(fig, Figure)

    # catch failure mode
    with pytest.raises(ValueError) as exc:
        plot.potential(0.1, h=0)
    assert str(exc.value) == ("Orbital angular momentum must exceed ISCO "
                              "(where h_ISCO**2 = 12)")


def test_diagnostic():
    """Test :func:`pyschild.orbit.plot.diagnostic`
    """
    y0 = timelike.initial_values(0.5, h=10)
    (geodesic, duration) = timelike.simulate(
        y0, timelike.HISCO, tf=1000)
    fig = plot.diagnostic(geodesic, duration)
    assert isinstance(fig, Figure)


def test_track():
    """Test :func:`pyschild.orbit.plot.track`
    """
    y0 = timelike.initial_values(0.5, h=10)
    (geodesic, duration) = timelike.simulate(
        y0, timelike.HISCO, tf=1000)
    fig = plot.track(geodesic, duration, title="Test track")
    assert isinstance(fig, Figure)
