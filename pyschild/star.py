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

"""Core utilities for simulating a field of stars
"""

import numpy
from numpy.random import rand

from astropy.io import registry
from astropy.table import Table
from astropy.units import Unit

from .sky import SkyMap

__author__ = "Alex Urban <alexander.urban@ligo.org>"

ARCMIN = float(Unit("arcmin").to("rad"))


# -- utilities ----------------------------------------------------------------

def _uniform_sphere(n):
    """Sample zenith and azimuth coordinates uniformly over the unit sphere

    Parameters
    ----------
    n : `int`
        total number of points to sample

    Returns
    -------
    theta, phi : `~numpy.ndarray`
        angular location samples on the unit sphere
    """
    theta = numpy.arccos(1 - 2 * rand(n))
    phi = 2 * numpy.pi * rand(n)
    return (theta, phi)


def _brightness_from_mag(mapp):
    """Infer visual brightness from apparent magnitude

    This is a convenience function to determine the Matplotlib color scale
    index from apparent bolometric magnitude, presuming objects fainter than
    magnitude 6.5 are not visible to the naked eye, while those brighter than
    magnitude 0 saturate the observer's viewing device.
    """
    too_dim = (mapp > 6.5)
    grad = numpy.logical_and(mapp <= 6.5, mapp > 0)
    saturated = (mapp <= 0)
    return numpy.piecewise(mapp, [too_dim, grad, saturated],
                           [0, lambda x: 1 - x / 6.5, 1])


def inherit_table_io(cls):
    """Inherit file I/O registrations from `~astropy.table.Table`

    This decorator is modeled on `~gwpy.table.inherit_io_registrations`,
    authored by Duncan Macleod, for more see https://gwpy.github.io
    """
    for row in registry.get_formats(data_class=Table):
        name = row["Format"]
        # read
        if row["Read"].lower() == "yes":
            registry.register_reader(
                name,
                cls,
                registry.get_reader(name, Table),
                force=False,
            )
        # write
        if row["Write"].lower() == "yes":
            registry.register_writer(
                name,
                cls,
                registry.get_writer(name, Table),
                force=False,
            )
        # identify
        if row["Auto-identify"].lower() == "yes":
            registry.register_identifier(
                name,
                cls,
                registry._identifiers[(name, Table)],
                force=False,
            )
    return cls


# -- classes ------------------------------------------------------------------

class Star(object):
    """Base class for star objects, including convenience methods for
    gravitational lensing and viewing under a point-spread function

    Parameters
    ----------
    theta : `float`, optional
        zenith coordinate (radians) of the star, default: `~numpy.pi / 2`

    phi : `float`, optional
        azimuth coordinate (radians) of the star, default: 0

    mag : `float`, optional
        apparent visual magnitude of the star, default: 0

    angrad : `float`, optional
        apparent angular radius (in radians) on the sky,
        default: 2.908e-4 radians (1 arcminute)

    brightness : `float`, optional
        normalized value between 0 (dim) and 1 (bright) representing the
        Matplotlib color gradient, default: 1
    """
    def __init__(self, theta=numpy.pi/2, phi=0,
                 mag=0, angrad=ARCMIN, brightness=1):
        """Initialize this `Star`
        """
        self.theta = theta
        self.phi = phi
        self.mag = mag
        self.brightness = brightness
        self.angrad = angrad

    def image(self, skymap, spread=3, copy=True):
        """Compose an image of this `Star` on a given sky map

        Parameters
        ----------
        skymap : `SkyMap`
            HEALPix map representing the entire sky seen by an observer

        spread : `int`, optional
            half-width (in multiples of `self.angrad`) of the pencil beam
            subtending this star's image, default: 3

        copy : `bool`, optional
            if `True`, copies the input sky map rather than populating
            in-place, default: `True`

        Returns
        -------
        out : `SkyMap`
            a version of ``sky`` with pixels populated representing
            this `Star` relative to the observer

        Notes
        -----
        This method represents the point-spread function of the receiver
        device as a simple Gaussian in ``self.theta`` and ``self.phi``,
        implicitly assuming the apparent angular size of the star is small
        enough that period boundary conditions can be neglected.

        See also
        --------
        StarField.sky
            a method that populates the sky for a fixed observer
        healpy
            a Python package to handle pixelated data on the unit sphere
        """
        if not isinstance(skymap, SkyMap):
            raise TypeError("Input must be an instance of SkyMap")
        out = skymap.copy() if copy else skymap
        # illuminate pixels within `spread` angular radii of the source
        image = out.pencil(self.theta, self.phi, spread * self.angrad)
        (theta, phi) = image.angles
        out[image.pindex] += self.brightness * numpy.exp(-0.5 * (
            ((theta - self.theta) / self.angrad)**2 +
            ((phi - self.phi) / self.angrad)**2
        ))
        return out


@inherit_table_io
class StarField(object):
    """Iterable representation of an all-sky field of stars, with convenience
    methods for representing star images on the sky

    By default, this class constructs a table of stars by sampling points
    uniformly over the unit sphere, then sampling apparent magnitudes and
    angular sizes from a uniform distribution.

    Alternatively, users can specify arrays of ``theta`` and ``phi``
    coordinates along with the visual properties, or read them from a table.

    Parameters
    ----------
    nstars : `int`, optional
        number of stars in the entire sky, default: `1e4`

    mdim : `float`, optional
        magnitude cutoff on the visually dim end, default: 6.5

    mbright : `float`, optional
        magnitude cutoff on the visually bright end, default: -2

    Alternatively, the following parameters may also be specified by-hand:

    theta : `~numpy.ndarray`, optional
        zenith coordinates (radians) of each star,
        required if ``phi`` is given

    phi : `~numpy.ndarray`, optional
        azimuth coordinates (radians) of each star,
        required if ``theta`` is given

    mag : `~numpy.ndarray`, optional
        apparent visual magnitudes of each star

    angrad : `~numpy.ndarray`, optional
        apparent angular radius (arcminutes) of each star on the sky

    brightness : `~numpy.ndarray`, optional
        normalized values between 0 (dim) and 1 (bright) representing the
        Matplotlib color gradient of each star

    Raises
    ------
    KeyError
        if either ``theta`` or ``phi`` are given without the other

    Example
    -------
    To simulate a field of stars with roughly the same number as are visible
    to the naked eye around Earth:

    >>> from pyschild import StarField
    >>> field = StarField()

    This allows for tabulation of each star's visible properties:

    >>> print(field.table)

    as well as for file I/O:

    >>> field.write('star-field.csv')

    Notes
    -----
    If individual attributes are specified directly, these will override
    the defaults.

    See also
    --------
    Star
        an object class designed to handle individual stars in the field
    """
    def __init__(self, nstars=1e4, mdim=6.5, mbright=-2, **kwargs):
        """Initialize this `StarField`
        """
        nstars = int(nstars)
        # angular coordinates
        try:
            self.theta = numpy.array(kwargs['theta'])
            self.phi = numpy.array(kwargs['phi'])
        except KeyError as exc:
            if ('theta' in kwargs) != ('phi' in kwargs):
                raise type(exc)("Both theta and phi arguments are required, "
                                "or draw from a uniform distribution")
            (self.theta, self.phi) = _uniform_sphere(nstars)
        # visual properties
        self.size = len(self.theta)
        self.mag = numpy.array(kwargs.pop(
            'mag',
            (mdim - mbright) * rand(self.size) + mbright))
        self.brightness = numpy.array(kwargs.pop(
            'brightness',
            _brightness_from_mag(self.mag)))
        self.angrad = numpy.array(kwargs.pop(
            'angrad',
            ARCMIN * (5 * rand(self.size) + 1)))

    def __iter__(self):
        """Iterate over this `StarField`

        Yields a `Star` for each object in the cluster
        """
        # yield a `Star` for each row
        for row in self.table:
            yield Star(row['theta'], row['phi'], row['mag'],
                       row['angrad'], row['brightness'])

    @property
    def table(self):
        """An iterable, tabular list of stars in the cluster
        """
        # record star properties
        return Table(
            [self.theta, self.phi, self.mag,
             self.angrad, self.brightness],
            names=('theta', 'phi', 'mag',
                   'angrad', 'brightness'),
        )

    # -- data I/O ------------------------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `StarField`

        This utility is essentially a wrapper around
        `~astropy.table.Table.read` which populates a `StarField` from the
        data collected in a `Table`

        Parameters
        ----------
        source : `str`
            path of a single data file from which to read

        *args : `tuple`, optional
            other positional arguments passed directly to the
            underlying reader method for the given format

        format : `str`, optional
            file format for input data; if not given, the data reader will
            attempt to automatically identify the format based on ``source``

        **kwargs : `dict`, optional
            other keyword arguments passed directly to the
            underlying reader method for the given format

        Returns
        -------
        field : `StarField`

        Raises
        ------
        astropy.io.registry.IORegistryError
            if the `format` cannot be automatically identified

        Notes
        -----"""
        # read as a table, then convert to a `StarField`
        data = registry.read(Table, source, *args, **kwargs)
        return cls(
            theta=data['theta'],
            phi=data['phi'],
            mag=data['mag'],
            angrad=data['angrad'],
            brightness=data['brightness'],
        )

    def write(self, target, *args, **kwargs):
        """Write the `StarField` table to a file

        This utility is essentially a wrapper around
        `~astropy.table.Table.write`, see that method for more information

        Parameters
        ----------
        target : `str`
            filename for output data file

        *args : `tuple`, optional
            other positional arguments passed to the underlying data writer
            for the given format

        format : `str`, optional
            file format for output data; if not given, the data writer will
            attempt to automatically identify the format based on ``target``

        **kwargs : `dict`, optional
            other keyword arguments passed to the underlying data writer
            for the given format

        Notes
        -----"""
        return registry.write(self.table, target, *args, **kwargs)

    # -- representation and visualization ------------------

    def sky(self, nside=2**12, nest=True, info="All-sky mock star field"):
        """Represent this `StarField` as a fixed observer's sky

        Returns a map of the sky in Hierarchical Equal Area isoLatitude
        Pixelization (HEALPix) format

        Parameters
        ----------
        nside : `int`, optional
            healpix nside parameter, must be a power of 2 less than ``2**30``,
            default: ``2**12`` or 4096

        nest : `bool`, optional
            if `True`, assumes NESTED pixel ordering, otherwise uses
            RING pixel ordering, default: `True`

        info : `str`, optional
            brief description of the output sky map, e.g.
            "All-sky mock star field" (the default)

        Returns
        -------
        out : `SkyMap`
            a pixelated image of the entire sky represented in HEALPix format

        Notes
        -----
        The default value ``nside=4096`` corresponds to an angular resolution
        per pixel of 0.859 arcminutes or 51.5 arcseconds, which approximates
        the average daytime spatial resolution of the human eye.

        By default this method returns a HEALPix map in the NESTED pixel
        ordering scheme, which optimizes any computation involving
        nearest-neighbor searches. By contrast, the RING scheme would
        optimize computations involving spherical harmonic transforms,
        for more see https://healpix.sourceforge.io/pdf/intro.pdf.

        See also
        --------
        Star.image
            the method which populates an image of a given star on the sky

        References
        ----------
        [1] Healpy, a Python package to handle pixelated data on the unit
            sphere: https://healpy.readthedocs.io/en/latest/index.html

        [2] The NASA HEALPix format specification,
            https://healpix.jpl.nasa.gov
        """
        nside = int(nside)
        # populate a map with star images
        out = SkyMap(
            numpy.zeros(12 * nside**2),
            info=info,
            nest=nest,
        )
        for star in self:
            star.image(out, copy=False)
        # return a `SkyMap` with saturated pixels
        return out.saturate()
