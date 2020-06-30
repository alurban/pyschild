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

"""Tests for :mod:`pyschild.orbit.__main__`

..note: These tests do not check that the output is correct, only that
        data products are created and plotting utilities run to completion.
"""

import os
import pytest
import shutil

from .. import __main__ as pyschild_orbit_cli

__author__ = "Alex Urban <alexander.urban@ligo.org>"


def test_main_with_help():
    args = ["--help"]
    with pytest.raises(SystemExit) as exc:
        pyschild_orbit_cli.main(args)
    assert exc.value.code == 0


def test_main(capsys, tmpdir):
    """Test :mod:`pyschild.orbit.__main__`
    """
    wdir = str(tmpdir)
    out = os.path.join(wdir, 'test')
    args = [
        "--output-dir", out,
        "--lorentz-factor", '1',
        "--angular-momentum", '5',
    ]

    # run with reasonable settings
    exitcode = pyschild_orbit_cli.main(args)
    (_, err) = capsys.readouterr()
    assert not err  # no warnings or error messages
    assert not exitcode  # passed

    # check that files exist
    assert os.path.isfile(os.path.join(out, "potential-well.png"))
    assert os.path.isfile(os.path.join(out, "figures-of-merit.png"))
    assert os.path.isfile(os.path.join(out, "orbital-track.png"))
    assert os.path.isfile(os.path.join(out, "orbital-trajectory.h5"))

    # overwrite files on re-run
    exitcode = pyschild_orbit_cli.main(args)
    assert not exitcode

    # clean up
    shutil.rmtree(wdir, ignore_errors=True)
