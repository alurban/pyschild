# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""A python package for visualizing the Schwarzschild spacetime

PySchild is an education-focused `Python <http://www.python.org>`_ package
providing visualization tools for simple, stationary black hole spacetimes.
It is intended for use by amateur science enthusiasts, undergraduates, and
graduate students alike, with easy-to-follow tutorials at every step.
"""

from ._version import get_versions

from .sky import SkyMap
from .star import (Star, StarField)

# set package metadata
__version__ = get_versions()['version']
__author__ = "Alex Urban <alexander.urban@ligo.org>"
__all__ = ['SkyMap', 'Star', 'StarField']

del get_versions
