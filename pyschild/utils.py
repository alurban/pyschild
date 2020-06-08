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

"""Misc. utilities and convenience functions
"""

import numpy

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- utilities ----------------------------------------------------------------

def power_sample(start, stop, num=50, base=2, **kwargs):
    """Convenience function to sample values clustered by a power law

    Parameters
    ----------
    start : `array_like`
        starting value of the sequence

    stop : `array_like`
        end value of the sequence, unless ``endpoint`` is set to `False`,
        for details see `~numpy.linspace`

    num : `int`, optional
        number of samples to generate, must be non-negative, default: 50

    base : `int` or `float`, optional
        base of power law to sample within, default: 2

    **kwargs
        additional keyword arguments to `~numpy.linspace`

    Returns
    -------
    samples : `~numpy.ndarray`
        ``num`` equally spaced samples in either the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        depending on whether ``endpoint`` is `True` or `False`

    See also
    --------
    numpy.linspace
        a utility to uniformly sample points over a specified interval
    numpy.power
        a utility to raise array elements to a power
    """
    return start + (stop - start) * (1 - numpy.power(
        numpy.linspace(0, 1, num=num, retstep=False, **kwargs),
        base,
    )[::-1])


def zero_crossings(x):
    """Convenience function to count zero-crossings in an array

    Parameters
    ----------
    x : `array_like`
        array of values within which to count zero-crossings

    Returns
    -------
    zeros : `~numpy.ndarray`
        array of indices where ``x`` has a zero-crossing
    """
    sign = numpy.sign(x)
    diff = numpy.diff(sign)
    (cross, ) = numpy.where(diff)
    return cross
