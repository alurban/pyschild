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

from numpy.linalg import norm

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- utilities ----------------------------------------------------------------

def get_rotation(direction):
    """Convenience function to get the angle and axis of a rotation

    Parameters
    ----------
    direction : `array_like`
        3-vector defining the direction along which to
        re-orient the x-axis

    Returns
    -------
    axis : `~numpy.ndarray`
        unit 3-vector defining the axis of rotation

    angle : `float`
        rotation angle (radians), lies within range ``[0, pi]``
    """
    reference = numpy.array((1, 0, 0))
    direction = numpy.asarray(direction) / norm(direction)
    # non-rotation
    if numpy.allclose(direction, reference):
        return (reference, 0)
    # under-determined rotation
    elif numpy.allclose(direction, -reference):
        return (numpy.array((0, 0, 1)), numpy.pi)
    # general rotation
    axis = numpy.cross(
        reference,
        direction,
    )
    angle = numpy.arccos(
        numpy.dot(
            reference,
            direction,
        ),
    )
    return (axis, angle)


def rotate(vec, angle, axis):
    """Rotate 3-vector(s) through an angle about some axis

    Parameters
    ----------
    vec : `array_like`
        Cartesian 3-vector(s) to rotate, must have shape either ``(3, )``
        for a single vector or ``(N, 3)`` for a collection of N vectors

    angle : `float`
        rotation angle (radians)

    axis : `array_like`
        any 3-vector co-aligned with the axis around which to rotate

    Returns
    -------
    rotated : `~numpy.ndarray`
        rotated versions of the input ``vec``

    Notes
    -----
    This utility obeys the right-hand rule, so rotations about ``axis`` are
    counter-clockwise. For clockwise rotation, pass the negative of either
    ``angle`` or ``axis`` (but not both) to reverse the orientation.

    See also
    --------
    pyschild.SkyMap.rotate
        method to re-orient a `SkyMap` instance along a given direction
    """
    (x, y, z) = numpy.asarray(vec).T
    axis = numpy.asarray(axis) / norm(axis)
    # special matrices
    ident = numpy.identity(3)
    levciv = numpy.zeros((3, 3, 3))
    levciv[0, 1, 2] = levciv[1, 2, 0] = levciv[2, 0, 1] = 1
    levciv[0, 2, 1] = levciv[2, 1, 0] = levciv[1, 0, 2] = -1
    # construct and apply rotation matrix
    rot = (numpy.cos(angle) * ident +
           (1 - numpy.cos(angle)) * numpy.outer(axis, axis) -
           numpy.sin(angle) * (levciv @ axis))
    return numpy.array((
        rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z,
        rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z,
        rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z,
    )).T


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
        ``num`` power-law spaced samples in either the closed interval
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
    """Convenience function to find zero-crossings in an array

    Parameters
    ----------
    x : `array_like`
        array of values within which to find zero-crossings

    Returns
    -------
    zeros : `~numpy.ndarray`
        array of indices where ``x`` has a zero-crossing
    """
    sign = numpy.sign(x)
    diff = numpy.diff(sign)
    (cross, ) = numpy.where(diff)
    return cross
