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

import os
import numpy
import pathlib
import pytest
import shutil

from numpy.testing import assert_array_equal

from ..signal import (read_stereo, write_stereo, Signal)

__author__ = "Alex Urban <alexander.urban@ligo.org>"

PARENT = pathlib.Path(__file__).parent.absolute()
LEFT_SIGNAL = Signal(numpy.ones(64), sample_rate=64)
RIGHT_SIGNAL = Signal(numpy.ones(64), sample_rate=64)


# -- test I/O utilities -------------------------------------------------------

def test_read_stereo():
    """Test `pyschild.signal.read_stereo
    """
    source = os.path.join(PARENT, 'data/stereo-signal.wav')
    (left, right) = read_stereo(source)
    assert isinstance(left, Signal)
    assert isinstance(right, Signal)
    assert left.is_compatible(right)
    assert_array_equal(left.value, LEFT_SIGNAL.value)
    assert_array_equal(right.value, RIGHT_SIGNAL.value)


def test_write_stereo(tmpdir):
    """Test `pyschild.signal.write_stereo`
    """
    base = str(tmpdir)
    write_stereo(  # test functional output
        os.path.join(base, 'stereo-signal.wav'),
        LEFT_SIGNAL, RIGHT_SIGNAL)
    with pytest.raises(ValueError) as exc:
        RIGHT_SIGNAL.sample_rate = 128
        write_stereo(  # test failure mode
            os.path.join(base, 'stereo-failure.wav'),
            LEFT_SIGNAL, RIGHT_SIGNAL)
    assert str(exc.value) == ("Signal arrays must have the "
                              "same length and sample rate")
    shutil.rmtree(base, ignore_errors=True)


# -- test time dilation methods -----------------------------------------------

class TestSignal(object):
    """Test `pyschild.signal.Signal`
    """
    TEST_CLASS = Signal
    TEST_SIGNAL = LEFT_SIGNAL

    def test_dilate(self):
        """Test `pyschild.signal.Signal.dilate`
        """
        dilated = self.TEST_SIGNAL.dilate(2, inplace=False)
        assert self.TEST_SIGNAL.is_compatible(dilated)
        assert (dilated.duration.value ==
                2 * self.TEST_SIGNAL.duration.value)
        assert (dilated.sample_rate.value ==
                self.TEST_SIGNAL.sample_rate.value)
        assert_array_equal(dilated.value, numpy.ones(dilated.size))

    def test_warp(self):
        """Test `pyschild.signal.Signal.warp`
        """
        warped = self.TEST_SIGNAL.warp(2)
        assert self.TEST_SIGNAL.is_compatible(warped)
        assert (warped.duration.value ==
                2 * self.TEST_SIGNAL.duration.value)
        assert (warped.sample_rate.value ==
                self.TEST_SIGNAL.sample_rate.value)
        assert_array_equal(warped.value, numpy.ones(warped.size))
