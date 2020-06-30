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

"""Unit tests for :mod:`pyschild.cli`
"""

import logging
import argparse

import pytest

from .. import (cli, __version__ as pyschild_version)

__author__ = "Alex Urban <alexander.urban@ligo.org>"


# -- test logger --------------------------------------------------------------

def test_logger():
    logger = cli.logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'pyschild.cli'


# -- test CLI -----------------------------------------------------------------

@pytest.fixture
def parser():
    return argparse.ArgumentParser()


def test_create_parser():
    parser = cli.create_parser(description=__doc__)
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description == __doc__
    assert parser._actions[-1].version == pyschild_version


@pytest.mark.parametrize('inv, outv', [
    (None, pyschild_version),
    ('test', 'test'),
])
def test_add_version_option(parser, inv, outv):
    act = cli.add_version_option(parser, version=inv)
    assert act.version == outv
    with pytest.raises(SystemExit):
        parser.parse_args(['--version'])
