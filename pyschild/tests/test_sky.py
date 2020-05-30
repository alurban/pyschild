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

from numpy.testing import assert_array_equal

from astropy import units

from ..sky import SkyMap

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

        # test with some data
        skymap = self.TEST_SKY_MAP
        assert isinstance(skymap, self.TEST_CLASS)
        assert_array_equal(skymap.value, self.data)

        # test with view casting
        skycast = self.data.view(self.TEST_CLASS)
        assert isinstance(skycast, self.TEST_CLASS)
        assert_array_equal(skycast.value, self.data)

        # test that copy=True ensures owndata
        assert self.create(copy=False).flags.owndata is False
        assert self.create(copy=True).flags.owndata is True

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
        indices = numpy.arange(2, 40)
        subset_map = skymap[indices]
        assert isinstance(subset_map, self.TEST_CLASS)
        assert subset_map.partial
        assert_array_equal(skymap.value[indices], subset_map.value)
        assert_array_equal(skymap.pindex[indices], subset_map.pindex)

    def test_attributes(self):
        """Test basic, default attributes are set on creation
        """
        skymap = self.TEST_SKY_MAP
        assert skymap.info == ""
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
