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

"""Tests for :mod:`pyschild.orbit.timelike`
"""

import numpy
import pytest

from numpy.testing import assert_allclose
from scipy.signal import (argrelmin, argrelmax)

from .. import timelike

__author__ = "Alex Urban <alexander.urban@ligo.org>"

ANGMOM = 4
R = numpy.arange(3, 100, 0.1)


# -- test utilities -----------------------------------------------------------

def count_zero_crossing(x):
    """Convenience function to count zero-crossings in an array
    """
    sign = numpy.sign(x)
    diff = numpy.diff(sign)
    (cross, ) = numpy.where(diff)
    return cross.size


def test_radial_potential():
    """Test :func:`pyschild.orbit.timelike.radial_potential`
    """
    # test with no angular momentum
    p1 = timelike.radial_potential(R, 0)
    assert count_zero_crossing(numpy.diff(p1)) == 0

    # test with critical angular momentum
    p2 = timelike.radial_potential(R, timelike.HISCO)
    (pmin2, ) = argrelmin(numpy.diff(p2))
    (pmax2, ) = argrelmax(numpy.diff(p2))
    assert_allclose(R[pmin2], 6)
    assert_allclose(R[pmax2], 12)

    # test with angular momentum that supports stable circular orbits
    p3 = timelike.radial_potential(R, ANGMOM)
    (pmin3, ) = argrelmin(p3)
    (pmax3, ) = argrelmax(p3)
    assert pmin3.size == 1
    assert pmax3.size == 1
    assert not numpy.array_equal(R[pmin3], R[pmax3])


def test_radial_force():
    """Test :func:`pyschild.orbit.timelike.radial_force`
    """
    # test with no angular momentum
    f1 = timelike.radial_force(R, 0)
    assert count_zero_crossing(f1) == 0

    # test with critical angular momentum
    f2 = timelike.radial_force(R, timelike.HISCO)
    (fmin, ) = argrelmin(f2)
    (fmax, ) = argrelmax(f2)
    assert_allclose(R[fmax], 6)
    assert_allclose(R[fmin], 12)

    # test with angular momentum that supports stable circular orbits
    f3 = timelike.radial_force(R, ANGMOM)
    assert count_zero_crossing(f3) == 2


def test_initial_values():
    """Test :func:`pyschild.orbit.timelike.initial_values`
    """
    # test at ISCO
    (r, rdot, phi) = timelike.initial_values(0)
    assert_allclose(r, 6)
    assert_allclose(rdot, 0)
    assert_allclose(phi, numpy.pi / 2)

    # test failure mode
    with pytest.raises(ValueError) as exc:
        timelike.initial_values(0, h=0)
    assert str(exc.value) == ("Orbital angular momentum must exceed ISCO "
                              "(where h_ISCO**2 = 12)")


def test_rhs():
    """Test :func:`pyschild.orbit.timelike.rhs`
    """
    y0 = timelike.initial_values(0)
    rhs = timelike.rhs(0, y0, timelike.HISCO)
    assert_allclose(rhs, [0, 0, timelike.HISCO / 36], atol=1e-16)


def test_simulate(capsys):
    """Test :func:`pyschild.orbit.timelike.simulate`
    """
    # test at ISCO
    y0 = timelike.initial_values(0)
    isco = timelike.simulate(y0, timelike.HISCO, verbose=True)
    stdout = capsys.readouterr().out
    assert stdout[:-1] == timelike.VERBOSE.format(
        code=0,
        nfev=26,
        status=0,
        message=("The solver successfully reached the end "
                 "of the integration interval."),
    )
    times = numpy.arange(60)
    (r, rdot, phi) = isco(times)
    dphi = numpy.diff(phi)
    assert_allclose(r, 6)
    assert_allclose(rdot, 0, atol=1e-15)
    assert_allclose(dphi, dphi[0])

    # test a failure mode
    with pytest.raises(RuntimeError) as exc:
        timelike.simulate(
            timelike.initial_values(0.2),
            timelike.HISCO,
            tf=1000,
        )
    assert str(exc.value) == ("Required step size is less than "
                              "spacing between numbers.")
