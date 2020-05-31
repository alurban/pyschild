#!/usr/bin/env python
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

"""Parse a requirements.txt-format file for use with Conda
"""

import argparse
import atexit
import json
import os
import pkg_resources
import re
import subprocess
import sys
import tempfile

from distutils.spawn import find_executable

__author__ = "Alex Urban <alexander.urban@ligo.org"
__credits__ = "Duncan Macleod <duncan.macleod@ligo.org"

# global variables
CONDA = (find_executable("conda") or
         os.environ.get("CONDA_EXE", "conda"))
CONDA_PACKAGE_MAP = {
    "matplotlib": "matplotlib-base"
}
(_, TMPFILE) = tempfile.mkstemp()
VERSION_OPERATOR = re.compile('[><=!]')


# -- utilities ----------------------------------------------------------------

def parse_requirements(file_):
    """Parse a pip requirements file
    """
    for line in file_:
        if line.startswith("-r "):
            name = line[3:].rstrip()
            with open(name, "r") as file2:
                yield from parse_requirements(file2)
        else:
            yield from pkg_resources.parse_requirements(line)


def _clean():
    """Clean up temporary files from the local system
    """
    if os.path.isfile(TMPFILE):
        os.remove(TMPFILE)


# -- parse requirements -------------------------------------------------------

atexit.register(_clean)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('filename', help='path of requirments file to parse')
parser.add_argument('-p', '--python-version',
                    default='{0.major}.{0.minor}'.format(sys.version_info),
                    help='python version to use (default: %(default)s)')
parser.add_argument('-o', '--output', help='path of output file, '
                                           'defaults to stdout')
args = parser.parse_args()

requirements = ["python={0.python_version}.*".format(args)]
with open(args.filename, "r") as reqf:
    for item in parse_requirements(reqf):
        # if environment markers don't pass, skip
        if item.marker and not item.marker.evaluate():
            continue
        # if requirement is a URL, skip
        if item.url:
            continue
        name = CONDA_PACKAGE_MAP.get(item.name, item.name)
        requirements.append('{}{}'.format(name, item.specifier))

# print requirements to temp file
with open(TMPFILE, 'w') as reqfile:
    for req in requirements:
        print(req, file=reqfile)

# find all packages with conda
cmd = [CONDA,
       'install',
       '--quiet',
       '--dry-run',
       '--file', TMPFILE,
       '--json']
pfind = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    shell=False,
)
(stdout, _) = pfind.communicate()

if pfind.returncode:
    if isinstance(stdout, bytes):
        stdout = stdout.decode('utf-8')
    try:
        missing = [pkg.split('[', 1)[0].lower() for
                   pkg in json.loads(stdout)['packages']]
    except json.JSONDecodeError:
        # run again so it fails out in the open
        subprocess.check_call(cmd, shell=False)
        raise
    requirements = [
        req for req in requirements if
        VERSION_OPERATOR.split(req)[0].strip().lower() not in missing]

# print output to file or stdout
if args.output:
    fout = open(args.output, 'w')
else:
    fout = sys.stdout
try:
    for req in requirements:
        print(req, file=fout)
finally:
    fout.close()
