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

"""Unit tests for :mod:`pyschild.sky`
"""

import os
import healpy
import numpy
import pathlib
import pytest
import shutil

from numpy.testing import (assert_array_equal,
                           assert_raises)

from astropy import units

from .. import SkyMap

__author__ = "Alex Urban <alexander.urban@ligo.org>"

PARENT = pathlib.Path(__file__).parent.absolute()
SEED = 4


class TestSkyMap(object):
    """Test `pyschild.sky.SkyMap
    """
    TEST_CLASS = SkyMap
    NSIDE = 2
    NPIX = 12 * NSIDE**2
    DTYPE = None

    # -- setup ----------------------------------

    @classmethod
    def setup_class(cls):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(cls.NPIX)).astype(dtype=cls.DTYPE)

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.setdefault('copy', False)
        return cls.TEST_CLASS(cls.data, *args, **kwargs)

    @classmethod
    @pytest.fixture()
    def array(cls):
        return cls.create()

    @property
    def TEST_SKY_MAP(self):
        try:
            return self._TEST_SKY_MAP
        except AttributeError:
            # create array
            self._TEST_SKY_MAP = self.create(dtype=self.DTYPE)
            return self.TEST_SKY_MAP

    # -- test basic construction ----------------

    def test_new(self):
        """Test `SkyMap` creation, including view casting
        """
        # test basic empty contructor
        with pytest.raises(TypeError):
            self.TEST_CLASS()

        # test non-trivial empty constructor
        empty = self.TEST_CLASS([])
        assert empty.size == 0
        assert empty.nside == 0
        with pytest.raises(ZeroDivisionError):  # zero-size pixels
            empty.area
        with pytest.raises(ValueError):  # invalid nside
            empty.directions

        # test with invalid shape
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS([[0], [1]])
        assert str(exc.value) == ("Only scalar or 1-dimensional data arrays "
                                  "are supported")

        # test with some data
        skymap = self.TEST_SKY_MAP
        assert isinstance(skymap, self.TEST_CLASS)
        assert_array_equal(skymap.value, self.data)

        # test view casting
        skycast = self.data.view(self.TEST_CLASS)
        assert isinstance(skycast, self.TEST_CLASS)
        assert_array_equal(skycast.value, self.data)

        newsky = skymap.view(self.TEST_CLASS)
        assert isinstance(newsky, self.TEST_CLASS)
        assert_array_equal(newsky.value, self.data)

        # test creation from template
        skyslice = skymap[20::2]
        assert not (skyslice is skymap)
        for attr in skymap._metadata_slots:
            val = getattr(skymap, attr)
            if val is None:
                assert val is getattr(skyslice, attr)
            elif isinstance(val, numpy.ndarray):
                assert_raises(AssertionError,
                              assert_array_equal,
                              val,
                              getattr(skyslice, attr))
            else:
                assert val == getattr(skyslice, attr)

        # test that copy=True ensures owndata
        assert self.create(copy=False).flags.owndata is False
        assert self.create(copy=True).flags.owndata is True

    def test_del(self):
        """Test deletion and re-construction of `_metadata_slots` properties
        """
        skymap = self.TEST_SKY_MAP

        # delete twice to test two logical pathways
        del skymap.info
        del skymap.info
        del skymap.nest
        del skymap.nest
        del skymap.nside
        del skymap.nside
        del skymap.pindex
        del skymap.pindex

        # ensure these attributes return to their default values
        assert skymap.info is None
        assert not skymap.nest
        assert skymap.nside == self.NSIDE
        assert_array_equal(skymap.pindex, numpy.arange(self.NPIX))

    def test_repr(self):
        """Test representations of `SkyMap` instances
        """
        skymap = self.TEST_SKY_MAP
        string = str(skymap)
        repres = repr(skymap)
        assert string.startswith('SkyMap(')
        assert string.endswith(')')
        assert "info: None," in string
        assert "nest: False," in string
        assert "nside: 2," in string
        assert repres.startswith('<SkyMap(')
        assert repres.endswith(')>')
        assert "info=None," in repres
        assert "nest=False," in repres
        assert "nside=2," in repres

    def test_deepcopy(self):
        """Test that a deep copy of `SkyMap` is also a `SkyMap`
        """
        from copy import deepcopy
        skymap = self.TEST_SKY_MAP
        cskymap = deepcopy(skymap)
        assert isinstance(cskymap, self.TEST_CLASS)
        assert cskymap.flags.owndata is True

    def test_math(self):
        """Test arithmetic operations on `SkyMap` instances
        """
        skymap = self.TEST_SKY_MAP

        # addition
        skymap1 = skymap + numpy.zeros(self.NPIX)
        assert isinstance(skymap1, self.TEST_CLASS)
        assert_array_equal(skymap.value, skymap1.value)

        # subtraction
        skymap2 = skymap - numpy.zeros(self.NPIX)
        assert isinstance(skymap2, self.TEST_CLASS)
        assert_array_equal(skymap.value, skymap2.value)

        # multiplication
        skymap3 = 3 * skymap
        assert isinstance(skymap3, self.TEST_CLASS)
        assert_array_equal(3 * skymap.value, skymap3.value)

    def test_subset(self):
        """Test that array subsets are also `SkyMap` instances
        """
        skymap = self.TEST_SKY_MAP
        slice_ = slice(2, 40, 3)
        subset_map = skymap[slice_]
        assert isinstance(subset_map, self.TEST_CLASS)
        assert subset_map.partial
        assert subset_map.nside == skymap.nside
        assert_array_equal(subset_map.value, skymap.value[slice_])
        assert_array_equal(subset_map.pindex, skymap.pindex[slice_])

    def test_attributes(self):
        """Test basic, default attributes are set on creation
        """
        skymap = self.TEST_SKY_MAP
        assert skymap.info is None
        assert (not skymap.nest)
        assert skymap.nside == self.NSIDE
        assert skymap.npix == self.NPIX
        assert_array_equal(skymap.pindex, numpy.arange(self.NPIX))
        assert (not skymap.partial)
        assert isinstance(skymap.explicit, dict)
        assert len(skymap.explicit) == self.NPIX
        assert skymap.resolution.value == healpy.nside2resol(
            self.NSIDE, arcmin=True)
        assert skymap.resolution.unit == units.arcmin
        assert skymap.pixrad.value == healpy.max_pixrad(
            self.NSIDE) * units.rad.to("arcmin")
        assert skymap.pixrad.unit == units.arcmin
        assert skymap.pixarea.value == healpy.nside2pixarea(
            self.NSIDE, degrees=True)
        assert skymap.pixarea.unit == units.deg ** 2
        assert skymap.area.value == skymap.size * skymap.pixarea.value
        assert skymap.area.unit == units.deg ** 2

    # -- test methods ---------------------------

    def test_read(self):
        """Test `SkyMap.read`
        """
        source = os.path.join(PARENT, 'data/complete-skymap.fits')
        cmap = self.TEST_CLASS.read(source)
        assert isinstance(cmap, self.TEST_CLASS)
        assert cmap.npix == cmap.size

    def test_write(self, tmpdir):
        """Test `SkyMap.write`
        """
        base = str(tmpdir)
        indices = numpy.arange(2, 40, 2)
        skymap = self.TEST_SKY_MAP
        skymap.write(os.path.join(base, 'complete-skymap.fits'))
        skymap[indices].write(os.path.join(base, 'partial-skymap.fits'))
        shutil.rmtree(base, ignore_errors=True)

    def test_nonzero(self):
        """Test `SkyMap.nonzero`
        """
        skymap = self.TEST_CLASS(
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        )
        nzmap = skymap.nonzero()
        assert nzmap.value.min() > 0
        assert nzmap.size <= nzmap.npix
        assert_array_equal(nzmap.pindex, skymap.value.nonzero()[0])

    def test_saturate(self):
        """Test `SkyMap.saturate`
        """
        skymap = 2 * self.TEST_SKY_MAP
        saturated = skymap.saturate(inplace=False)
        assert skymap.value.max() != 1
        assert saturated.value.max() == 1
        assert skymap.size == saturated.size
        assert skymap.npix == saturated.npix
        assert skymap.nside == saturated.nside

    def test_pencil(self):
        """Test `SkyMap.pencil`
        """
        skymap = self.TEST_SKY_MAP
        pbeam = skymap.pencil(0, 0, 1 * units.rad)
        assert pbeam.size < skymap.size
        assert_array_equal(
            pbeam.pindex,
            healpy.query_disc(
                skymap.nside,
                (0, 0, 1),
                1,
            ),
        )
