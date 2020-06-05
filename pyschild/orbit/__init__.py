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

"""Submodule for simulations of black hole orbits
"""

from astropy.constants import (
    G,
    M_sun,
    c,
)

__author__ = "Alex Urban <alexander.urban@ligo.org>"

# spacetime distances
RSOL = G * M_sun / c**2
TSOL = G * M_sun / c**3
