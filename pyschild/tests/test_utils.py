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

"""Unit tests for :mod:`pyschild.signal`
"""

import numpy

from numpy.testing import assert_array_equal

from .. import utils

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- test utilities -----------------------------------------------------------

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
