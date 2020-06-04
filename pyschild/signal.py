# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2020)
#
# This file is part of the PySchild python package.
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

"""Core utilities for manipulating time-domain signals
"""

import numpy
from scipy.io import wavfile

from gwpy.timeseries import TimeSeries

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- utilities ----------------------------------------------------------------

def read_stereo(source, mmap=False):
    """Read `Signal` objects from a WAV file (.wav) in stereo

    Parameters
    ----------
    source : `str`
        path to input WAV file

    mmap : bool, optional
        whether to read data as memory-mapped, only to be used on real files,
        default: `False`

    Returns
    -------
    tsleft, tsright : `Signal`
        a tuple of `Signal` objects with data and sample rates read from
        the source file. `tsleft` corresponds to the left-hand side data
        stream, `tsright` corresponds to the right-hand side

    See also
    --------
    scipy.io.wavfile.read
        the utility which directly reads WAV files
    gwpy.timeseries.TimeSeries.read
        an alternative class method for single-stream (i.e., mono) data
    """
    (sample, data) = wavfile.read(source)
    tsleft = Signal(data[:, 0], sample_rate=sample)
    tsright = Signal(data[:, 1], sample_rate=sample)
    return (tsleft, tsright)


def write_stereo(target, tsleft, tsright):
    """Write `Signal` objects to a WAV file (.wav) in stereo

    Parameters
    ----------
    target : `str`
        path to output WAV file

    tsleft : `Signal`
        the left-hand side data stream

    tsright : `Signal`
        the right-hand side data stream

    Raises
    ------
    ValueError
        if `tsleft` and `tsright` have inconsistent sizes or sample rates

    See also
    --------
    scipy.io.wavfile.write
        the utility which directly writes to WAV files
    gwpy.timeseries.TimeSeries.write
        an alternative class method for single-stream (i.e., mono) data
    """
    if (tsleft.dx != tsright.dx) or (tsleft.size != tsright.size):
        raise ValueError("Signal arrays must have the "
                         "same length and sample rate")
    sample = int(tsleft.sample_rate.to('Hz').value)
    data = numpy.array(list(zip(tsleft.value, tsright.value)))
    wavfile.write(target, sample, data)


# -- classes ------------------------------------------------------------------

class Signal(TimeSeries):
    """A time-domain representation of data

    This is a subclass of `~gwpy.timeseries.TimeSeries` with a couple of
    extra methods

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    dt : `float`, `~astropy.units.Quantity`, optional
        time between successive samples (seconds), can also be given inversely
        via `sample_rate`

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        the rate of samples per second (Hertz), can also be given inversely
        via `dt`

    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside

    name : `str`, optional
        descriptive title for this array

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional
        choose to copy the input data to new memory

    subok : `bool`, optional
        allow passing of sub-classes by the array generator

    See also
    --------
    gwpy.timeseries.TimeSeries
        for the parent class, a robust and powerful way to represent
        1-dimensional data in the time domain featuring a vast set of
        helpful methods
    """
    def dilate(self, factor, inplace=True):
        """Dilate this `Signal` in the time domain by a given factor

        Parameters
        ----------
        factor : `float`
            the ratio of elapsed apparent time (observer frame) to elapsed
            proper time (source frame)

        inplace : `bool`, optional
            whether to dilate this `Signal` in-place, i.e., without
            copying to a new object, default: `True`

        Returns
        -------
        out : `Signal`
            the time-dilated signal

        Notes
        -----
        This method first scales the sampling interval (``self.dt``) by
        ``factor``, then resamples the resulting data stream to the original
        sampling rate.
        """
        out = self if inplace else self.copy()
        out.dt *= factor
        # resample in the time domain
        return out.resample(self.sample_rate.to('Hz').value)

    def warp(self, factors):
        """Apply a sequence of time dilation factors to this `Signal`

        Parameters
        ----------
        factors : `~numpy.ndarray` of `float`
            collection of time dilation factors to apply over equal-sized
            chunks of this data

        Returns
        -------
        out : `Signal`
            a version of the input with time dilation applied

        Notes
        -----
        The positional argument ``factors`` should be a 1-dimensional array of
        length N, whence the input will be time-dilated in equal sized chunks
        up to approximate length ``self.size / N``.

        See also
        --------
        Signal.dilate
            a method that applies a uniform dilation in the time domain
        numpy.array_shift
            a utility for splitting one array into multiple sub-arrays
        """
        if not numpy.iterable(factors):
            factors = numpy.array([factors])
        out = Signal(
            numpy.array([]),  # append to an empty array
            sample_rate=self.sample_rate.to('Hz').value,
            t0=self.t0,
        )
        # pitch-shift signal samples
        for (chunk, factor) in zip(
                numpy.array_split(
                    numpy.arange(self.size),
                    numpy.size(factors)),
                factors,
        ):
            value = self[chunk].dilate(factor, inplace=False)
            value.t0 = (out.t0 + out.duration).value
            out.append(value)
        return out
