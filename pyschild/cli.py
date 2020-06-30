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

"""Command-line interface tools for `pyschild`
"""

import argparse
import coloredlogs
import datetime
import logging
import sys

from pytz import reference

from . import __version__

__author__ = "Alex Urban <alexander.urban@ligo.org>"

# logging variables
NOW = datetime.datetime.now()
TIMEZONE = reference.LocalTimezone().tzname(NOW)
DATEFMT = '%Y-%m-%d %H:%M:%S {}'.format(TIMEZONE)
FMT = '%(name)s %(asctime)s %(levelname)+8s: %(message)s'

LEVEL_STYLES = {
    'critical': {'color': 9, 'bold': True},
    'debug': {'color': 14},
    'error': {'color': 'red'},
    'info': {'color': 10},
    'notice': {'color': 'magenta'},
    'spam': {'color': 'green', 'faint': True},
    'success': {'color': 'green', 'bold': True},
    'verbose': {'color': 'blue'},
    'warning': {'color': 13},
}

FIELD_STYLES = {
    'levelname': {'color': 39},
    'asctime': {'color': 27},
    'name': {'color': 12},
}

# disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


# -- logging ------------------------------------------------------------------

def logger(name=__name__, level='DEBUG'):
    """Construct a logger utility for stdout/stderr messages
    """
    logger = logging.getLogger(name)
    coloredlogs.install(
        level=level, logger=logger, stream=sys.stdout, fmt=FMT,
        datefmt=DATEFMT, level_styles=LEVEL_STYLES, field_styles=FIELD_STYLES)
    return logger


# -- parsing ------------------------------------------------------------------

def create_parser(**kwargs):
    """Create a new `argparse.ArgumentParser`
    """
    version = kwargs.pop('version', __version__)
    parser = argparse.ArgumentParser(**kwargs)
    if version is not None:
        add_version_option(parser, version=version)
    return parser


def add_version_option(parser, version=None):
    if version is None:
        version = __version__
    return parser.add_argument('-V', '--version', action='version',
                               version=version)
