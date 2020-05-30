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

"""Unit tests for :mod:`pyschild.star`
"""

import os
import numpy
import pytest
import shutil

from numpy.testing import assert_array_equal

from astropy.units import Unit
from astropy.table import Table

from .. import (Star, StarField, SkyMap)

__author__ = "Alex Urban <alexander.urban@ligo.org>"

ARCMIN = float(Unit("arcmin").to("rad"))


class TestStar(object):
    """Test `pyschild.star.Star`
    """
    TEST_CLASS = Star

    # -- setup ----------------------------------

    @classmethod
    def create(cls, *args, **kwargs):
        return cls.TEST_CLASS(*args, **kwargs)

    @property
    def TEST_STAR(self):
        try:
            return self._TEST_STAR
        except AttributeError:
            # create a test object
            self._TEST_STAR = self.create()
            return self.TEST_STAR

    # -- test basic construction ----------------

    def test_new(self):
        """Test `Star` creation
        """
        # test with defaults
        star = self.TEST_STAR
        assert isinstance(star, self.TEST_CLASS)
        assert star.theta == numpy.pi / 2
        assert star.phi == 0
        assert star.mag == 0
        assert star.brightness == 1
        assert star.angrad == ARCMIN

    # -- test methods ---------------------------

    def test_image_failure(self):
        """Test the failure mode in `Star.image()`
        """
        skymap = numpy.zeros(12)
        with pytest.raises(TypeError) as exc:
            self.TEST_STAR.image(skymap, copy=False)
        assert str(exc.value) == "Input must be an instance of SkyMap"


class TestStarField(object):
    """Test `pyschild.star.StarField`
    """
    TEST_CLASS = StarField

    # -- setup ----------------------------------

    @classmethod
    def create(cls, *args, **kwargs):
        return cls.TEST_CLASS(*args, **kwargs)

    @property
    def TEST_STAR_FIELD(self):
        try:
            return self._TEST_STAR_FIELD
        except AttributeError:
            # create a test object
            self._TEST_STAR_FIELD = self.create()
            return self.TEST_STAR_FIELD

    # -- test basic construction ----------------

    def test_new(self):
        """Test `StarField` creation
        """
        # test case when only one angle is passed
        with pytest.raises(KeyError) as exc:
            self.create(theta=[0])
        assert str(exc.value).strip("'") == (
            "Both theta and phi arguments are required, "
            "or draw from a uniform distribution")

        # otherwise test with defaults
        field = self.TEST_STAR_FIELD
        assert isinstance(field, self.TEST_CLASS)
        assert field.size == int(1e4)
        assert numpy.logical_and(field.mag >= -2,
                                 field.mag <= 6.5).all()
        assert numpy.logical_and(field.brightness >= 0,
                                 field.brightness <= 1).all()

        # make sure attributes are all `~numpy.ndarray`
        for attr in (field.theta, field.phi, field.mag,
                     field.angrad, field.brightness):
            assert isinstance(attr, numpy.ndarray)
            assert attr.size == field.size

    def test_iter(self):
        """Test iterating through a `StarField`
        """
        for star in self.create(nstars=10):
            assert isinstance(star, Star)

    def test_table(self):
        """Test an attribute that tabulates the contents of `StarField`
        """
        field = self.TEST_STAR_FIELD
        table = field.table
        assert isinstance(table, Table)
        assert len(table) == field.size
        assert_array_equal(field.theta, table['theta'])
        assert_array_equal(field.phi, table['phi'])
        assert_array_equal(field.mag, table['mag'])
        assert_array_equal(field.angrad, table['angrad'])
        assert_array_equal(field.brightness, table['brightness'])

    # -- test methods ---------------------------

    def test_read(self):
        """Test `StarField.read`
        """
        # ensure docstring is successfully re-generated
        # to show supported file formats
        assert ("The available built-in formats are:"
                in self.TEST_CLASS.read.__doc__)

        # read and verify test data
        field = self.TEST_CLASS.read('data/star-field.csv')
        assert isinstance(field, self.TEST_CLASS)
        assert field.size == 50

    def test_write(self, tmpdir):
        """Test `StarField.write`
        """
        # ensure docstring is successfully re-generated
        # to show supported file formats
        assert ("The available built-in formats are:"
                in self.TEST_CLASS.read.__doc__)

        # check the writer works
        base = str(tmpdir)
        field = self.TEST_STAR_FIELD
        field.write(os.path.join(base, 'star-field.csv'))
        shutil.rmtree(base, ignore_errors=True)

    def test_sky(self):
        """Test `StarField.sky`
        """
        field = self.TEST_STAR_FIELD
        skymap = field.sky(nside=256)
        assert isinstance(skymap, SkyMap)
        assert skymap.nest
        assert skymap.nside == 256
        assert skymap.info == "All-sky mock star field"
        assert skymap.min().value == 0
        assert skymap.max().value == 1
