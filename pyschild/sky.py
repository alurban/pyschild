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

"""Core utilities for HEALPix sky map representation
"""

import numpy

import healpy
from astropy import units

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- classes ------------------------------------------------------------------

# FIXME: include spherical harmonic tools and rotations, then
#        write a ray tracer
class SkyMap(numpy.ndarray):
    """Representation of the sky for a fixed observer in Hierarchical Equal
    Area isoLatitude Pixelization (HEALPix) format

    Parameters
    ----------
    value : `~numpy.ndarray`
        HEALPix-ordered collection of numerical data on the sky

    nside : `int`, optional
        HEALPix ``nside`` parameter, required if the input is a partial sky
        map, default: infer from input data assuming complete coverage

    pindex : `~numpy.ndarray`, optional
        if given, maps pixel values in ``value`` to their corresponding
        pixel index, defaults to `~numpy.arange(value.size)`

    info : `str`, optional
        a brief description of this sky map, e.g. "All-sky map"
        default: `None`

    nest : `bool`, optional
        if `True`, assume NESTED pixel ordering, otherwise use
        RING pixel ordering, default: `False`

    dtype : `str`, optional
        `dtype` of the resulting Numpy array or scalar, e.g. `'float32'`
        default: infer from the input `value`

    copy : `bool`, optional
        if true (default) then the input object is copied, otherwise
        a copy will only be made if ``__array__`` returns a copy or if
        one is needed to satisfy any of the other requirements, such
        as `dtype`. This option is provided mainly for internal use
        to speed up initialization in cases where a known copy is already
        present, please use with care

    subok : `bool`, optional
        if `True`, sub-classes will be passed through, otherwise the
        returned array will be forced to a base-class array,
        default: `False`

    Raises
    -------
    ValueError
        if the ``value`` provided has more than one axis

    Example
    -------
    `SkyMap` objects are subclasses of `~numpy.ndarray` and can be created
    in the same way:

    >>> import numpy
    >>> from pyschild import SkyMap
    >>> skymap = SkyMap(numpy.zeros(12))

    This creates a (very coarse) pixelization of the sky populated with
    zeroes and a HEALPix ``nside`` parameter of 1:

    >>> print(skymap.nside)

    Users may also create partial sky maps, with the caveat that an explicit
    pixel index is also needed in this use case:

    >>> partial_skymap = SkyMap(numpy.zeros(3),
                                pindex=(0, 3, 7),
                                nside=1)

    Note, the ``nside`` parameter must be an integer power of 2, and for a
    complete all-sky map the number of pixels is `12 * nside**2`. Other
    parameters, such as the angular resolution, pixel area, and total area,
    are also available, e.g.:

    >>> print(skymap.resolution)
    >>> print(skymap.pixarea)
    >>> print(skymap.area)

    (For an all-sky map, ``skymap.area`` will amount ``4 * numpy.pi``.

    Notes
    -----
    This class definition assumes RING pixel ordering by default, mainly
    for consistency with healpy (https://healpy.readthedocs.io/en/latest/).
    The RING ordering scheme is optimal for operations involving spherical
    harmonic transforms, while the NESTED ordering scheme optimizes nearest
    neighbor searches, for more information see
    https://healpix.sourceforge.io/pdf/intro.pdf.

    See also
    --------
    numpy.ndarray
        the underlying superclass
    numpy.array
        a tool that casts any iterable object to `~numpy.ndarray`
    healpy
        a package designed to handle pixelated data on the unit sphere
    """
    _metadata_slots = ('info', 'nest', 'nside', 'npix', 'pindex')

    def __new__(cls, value, nside=None, pindex=None, info=None,
                nest=False, dtype=None, copy=True, subok=False):
        """Create a new `SkyMap`
        """
        # cast input to `numpy.ndarray`
        new = numpy.array(value, dtype=dtype, copy=False, subok=False)

        # view the array as `cls`
        new = new.view(cls)

        # explicitly copy here to get ownership of the data
        if copy:
            new = new.copy()

        # set new attributes
        new.info = info
        new.nest = nest
        new.nside = nside or healpy.npix2nside(new.size)
        new.npix = healpy.nside2npix(new.nside)
        new.pindex = (pindex if pindex is not None
                      else numpy.arange(new.size))
        return new

    def __array_finalize__(self, obj):
        # if a new object, do nothing
        if obj is None:
            return

        # if the array is not scalar or 1-dimensional, throw a fit
        if len(numpy.shape(obj)) > 1:
            raise ValueError("Only 1-dimensional data arrays are supported")

        # update metadata
        self.__metadata_finalize__(obj)

    def __metadata_finalize__(self, obj, force=False):
        # based on gwpy.types.array.Array.__metadata_finalize__,
        # credit: Duncan Macleod
        # apply metadata from obj to self if creating a new object
        for attr in self._metadata_slots:
            _attr = '_%s' % attr  # use private attribute (not property)
            # if attribute is unset, default it to `None`, then update
            # from obj if desired
            try:
                getattr(self, _attr)
            except AttributeError:
                update = True
            else:
                update = force
            if update:
                try:
                    val = getattr(obj, _attr)
                except AttributeError:
                    continue
                else:
                    setattr(self, _attr, val)

    def __deepcopy__(self, memo):
        # ensure ``copy.deepcopy`` does not return a bare NumPy array
        return self.copy()

    def __getitem__(self, key):
        # properly re-size indices when handling slices
        if isinstance(key, list):
            key = tuple(key)
        new = super().__getitem__(key)
        return type(self)(new, pindex=numpy.array(key), nside=self.nside,
                          info=self.info, nest=self.nest, dtype=self.dtype)

    def __add__(self, value):
        # handle the case when adding another array element-wise
        new = super().__add__(value)
        return type(self)(new, pindex=self.pindex, nside=self.nside,
                          info=self.info, nest=self.nest, dtype=self.dtype)

    def __sub__(self, value):
        # handle the case when subtracting another array element-wise
        new = super().__sub__(value)
        return type(self)(new, pindex=self.pindex, info=self.info,
                          nest=self.nest, dtype=self.dtype)

    # info
    @property
    def info(self):
        try:
            return self._info
        except AttributeError:
            self._info = ""
            return self._info

    @info.setter
    def info(self, val):
        self._info = ("" if val is None else str(val))

    @info.deleter
    def info(self):
        try:
            del self._info
        except AttributeError:
            pass

    # nest
    @property
    def nest(self):
        try:
            return self._nest
        except AttributeError:
            self._nest = False
            return self._nest

    @nest.setter
    def nest(self, val):
        self._nest = bool(val)

    @nest.deleter
    def nest(self):
        try:
            del self._nest
        except AttributeError:
            pass

    # nside
    @property
    def nside(self):
        try:
            return self._nside
        except AttributeError:
            self._nside = healpy.npix2nside(self.size)
            return self._nside

    @nside.setter
    def nside(self, val):
        self._nside = int(val)

    @nside.deleter
    def nside(self):
        try:
            del self._nside
        except AttributeError:
            pass

    # npix
    @property
    def npix(self):
        try:
            return self._npix
        except AttributeError:
            self._npix = healpy.nside2npix(self.nside)
            return self._npix

    @npix.setter
    def npix(self, val):
        self._npix = val

    @npix.deleter
    def npix(self):
        try:
            del self._npix
        except AttributeError:
            pass

    # pindex
    @property
    def pindex(self):
        try:
            return self._pindex
        except AttributeError:
            self._pindex = numpy.arange(self.size)[()]
            return self._pindex

    @pindex.setter
    def pindex(self, val):
        self._pindex = numpy.array(val)[()]

    @pindex.deleter
    def pindex(self):
        try:
            del self._pindex
        except AttributeError:
            pass

    # -- other properties ----------------------------------

    @property
    def partial(self):
        """Determine whether this `SkyMap` covers only part of the sky
        """
        return (self.size != self.npix)

    @property
    def value(self):
        """The raw numerical value of this `SkyMap` instance
        """
        return self.view(numpy.ndarray)[()]

    @property
    def explicit(self):
        """Represent `self.value` with explicit pixel indexing
        """
        key = (self.pindex if numpy.iterable(self.pindex)
               else numpy.array([self.pindex]))
        val = (self.value if numpy.iterable(self.value)
               else numpy.array([self.value]))
        return dict(zip(key, val))

    @property
    def resolution(self):
        """Angular resolution (arcminutes) of pixels in this `SkyMap`

        This returns an instance of `~astropy.units.Quantity` with explicit
        units, which can then be converted by the user as-needed.
        """
        return healpy.nside2resol(
            self.nside,
            arcmin=True,
        ) * units.Unit("arcmin")

    @property
    def pixrad(self):
        """Maximum angular distance (arcminutes) between any pixel center
        and its corners

        This returns an instance of `~astropy.units.Quantity` with explicit
        units, which can then be converted by the user as-needed.
        """
        radius = healpy.max_pixrad(self.nside) * units.Unit("rad")
        return radius.to("arcmin")

    @property
    def pixarea(self):
        """Solid area (square degrees) subtended by a pixel in this `SkyMap`

        This returns an instance of `~astropy.units.Quantity` with explicit
        units, which can then be converted by the user as-needed.
        """
        return healpy.nside2pixarea(
            self.nside,
            degrees=True,
        ) * units.Unit("deg") ** 2

    @property
    def area(self):
        """Solid area (square degrees) subtended by this `SkyMap`

        If the sky map is complete, i.e. if ``self.partial == False``, this
        property should essentially return ``4 * numpy.pi``.

        This returns an instance of `~astropy.units.Quantity` with explicit
        units, which can then be converted by the user as-needed.
        """
        return self.size * self.pixarea

    @property
    def angles(self):
        """Angles corresponding to pixels of this `SkyMap`

        Returns a tuple of ``(zenith, azimuth)``, where ``zenith`` is an array
        corresponding to zenith angles (more precisely, either latitude or
        colatitude) and ``azimuth`` is an array of azimuth (or longitude)
        angles, e.g.

        >>> (theta, phi) = skymap.angles

        These angles are an instance of `~astropy.units.Quantity` with
        explicit units, either degrees (for latitude/longitude) or radians
        (for zenith/azimuth).
        """
        return healpy.pix2ang(self.nside, self.pindex, nest=self.nest)

    @property
    def directions(self):
        """Vector components indicating direction from the origin to a point
        on the unit sphere, for each pixel in this `SkyMap`

        Returns a tuple of (unitless) Cartesian ``(x, y, z)`` components, e.g.

        >>> (x, y, z) = skymap.directions

        Because these are unit vectors, an element-wise quadrature sum will
        give unity, i.e. ``x**2 + y**2 + z**2`` is functionally equivalent to
        `~numpy.ones(self.size)`.
        """
        return healpy.pix2vec(self.nside, self.pindex, nest=self.nest)

    # -- data I/O ------------------------------------------

    @classmethod
    def read(cls, source, info="All-sky map", nest=False,
             dtype=numpy.float32, partial=False, verbose=False):
        """Read data from a FITS file into a `SkyMap`

        This method is essentially a wrapper around `~healpy.read_map`, see
        that utility for more information

        Parameters
        ----------
        source : `str`
            path of a single data file from which to read

        info : `str`, optional
            a brief description of this sky map, e.g.
            "All-sky map" (the default)

        nest : `bool`, optional
            if `True`, assume NESTED pixel ordering, otherwise use
            RING pixel ordering, default: `False`

        dtype : `str` or type class, optional
            `dtype` of the resulting Numpy array or scalar,
            default: `~numpy.float32`

        partial : `bool`, optional
            if `True`, FITS file is assumed to contain only part of the sky
            with explicit pixel indexing, if the indexing scheme cannot be
            determined from the header. If `False`, implicit indexing is
            assumed. Default: `False`

        verbose : `bool`, optional
            If `True`, prints a number of diagnostic messages,
            default: `False`

        Returns
        -------
        skymap : `SkyMap`

        Notes
        -----
        This method reads data files stored in Flexible Image Transport
        System (FITS) format, for more see https://fits.gsfc.nasa.gov.

        See also
        --------
        healpy.read_map
            the underlying FITS file reader
        """
        skymap = healpy.read_map(source, nest=nest, dtype=dtype,
                                 partial=partial, verbose=verbose)
        # unpack partial sky maps
        if partial or isinstance(skymap, dict):
            pindex = numpy.array([*skymap])
            skymap = numpy.array(list(skymap.values()))
        else:
            pindex = None
        return cls(skymap, pindex=pindex, info=info, nest=nest, dtype=dtype)

    def write(self, target, coord=None, column="BRIGHTNESS", overwrite=False):
        """Write this `SkyMap` to a FITS file

        Parameters
        ----------
        target : `str`
            filename for output data file

        coord : `str`, optional
            the coordinate system, must be one of `'E'` for Ecliptic,
            `'G'` for Galactic, or `'C'` for Celestial (equatorial),
            default: `None`

        column : `str`, optional
            all-caps name of the single data column of this sky map,
            default: `"BRIGHTNESS"`

        overwrite : `bool`, optional
            if `True`, silently overrides ``target`` if it already exists,
            else raises an `OSError`, default: `False`

        Raises
        ------
        OSError
            if ``overwrite=False`` and the target file already exists

        Notes
        -----
        This method writes data files in Flexible Image Transport System
        (FITS) format, for more see https://fits.gsfc.nasa.gov.

        See also
        --------
        healpy.write_map
            the underlying FITS file writer
        """
        data = self.explicit if self.partial else self.value
        healpy.write_map(target, data, coord=coord, nest=self.nest,
                         dtype=self.dtype, partial=self.partial,
                         column_names=[column], overwrite=overwrite)

    # -- analysis and visualization ------------------------

    def saturate(self, limit=1, inplace=True):
        """Saturate pixels in this `SkyMap` that exceed a given limit

        Parameters
        ----------
        limit : `float`, optional
            limiting value to saturate, default: 1

        inplace : `bool`, optional
            if `True`, perform operation in-place (modifying this `SkyMap`
            instance in memory), else return a copy, default: `True`

        Returns
        -------
        out : `SkyMap`
            a version of this instance whose peak value saturates at ``limit``
        """
        out = self if inplace else self.copy()
        out[self.value > limit] = limit
        return out

    def pencil(self, theta, phi, angrad, **kwargs):
        """Return a subset of this `SkyMap` that covers an angular disc on
        the sky (i.e., a pencil beam)

        Parameters
        ----------
        theta : `float`
            zenith angle (radians) at the center of the disc

        phi : `float`
            azimuth angle (radians) at the center of the disc

        angrad : `float` or `~astropy.units.Quantity`
            angular radius (radians) subtended by the disc

        **kwargs : `dict`, optional
            additional keyword arguments to `~healpy.query_disc`

        Returns
        -------
        out : `SkyMap`
            the subset of `SkyMap` subtended by this pencil beam
        """
        if isinstance(angrad, units.Quantity):
            angrad = angrad.to("rad").value
        direction = healpy.ang2vec(theta, phi)
        indices = healpy.query_disc(self.nside, direction, angrad,
                                    nest=self.nest, **kwargs)
        return self[indices]
