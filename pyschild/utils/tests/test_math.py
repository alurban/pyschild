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

"""Unit tests for :mod:`pyschild.utils.math`
"""

import numpy

from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

from .. import utils

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- test utilities -----------------------------------------------------------

def test_format_scientific():
    """Test :func:`pyschild.utils.format_scientific`
    """
    # raw float
    small = utils.format_scientific(1)
    assert small == '1.000'

    # large number
    large = utils.format_scientific(1e5)
    assert large == '1.000 \\times 10^{5}'


def test_get_rotation():
    """Test :func:`pyschild.utils.get_rotation`
    """
    # non-rotation
    (axis1, angle1) = utils.get_rotation((1, 0, 0))
    assert angle1 == 0
    assert_array_equal(axis1, (1, 0, 0))

    # under-determined rotation
    (axis2, angle2) = utils.get_rotation((-1, 0, 0))
    assert angle2 == numpy.pi
    assert_array_equal(axis2, (0, 0, 1))

    # more general rotation
    (axis3, angle3) = utils.get_rotation((0, 1, 0))
    assert_allclose(angle3, numpy.pi / 2)
    assert_allclose(axis3, (0, 0, 1))


def test_rotate():
    """Test :func:`pyschild.utils.rotate`
    """
    # single vector
    vec = (1, 0, 0)
    angle = numpy.pi / 2
    axis = (0, 0, 1)
    rotated = utils.rotate(vec, angle, axis)
    assert_allclose(rotated, (0, 1, 0), atol=1e-16)

    # array of vectors
    vec = ((0, -1, 0), (1, 0, 0))
    rotated = utils.rotate(vec, angle, axis)
    assert_allclose(rotated, ((1, 0, 0), (0, 1, 0)), atol=1e-16)


def test_power_sample():
    """Test :func:`pyschild.utils.power_sample`
    """
    samples = utils.power_sample(0, 1)
    assert_array_equal(samples, (1 - numpy.power(
        numpy.linspace(0, 1, num=50),
        2,
    ))[::-1])


def test_zero_crossings():
    """Test :func:`pyschild.utils.zero_crossings`
    """
    times = numpy.arange(0, 1, 1/256)
    signal = numpy.cos(2 * numpy.pi * times)
    crossings = utils.zero_crossings(signal)
    assert crossings.size == 2
    assert times[crossings[0]] == 0.25
    assert times[crossings[1]] == 0.75
