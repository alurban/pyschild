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

"""Setup the PySchild package
"""

from setuptools import (
    find_packages,
    setup,
)

# local module to handle build customisations
from setup_utils import (
    CMDCLASS,
    VERSION,
    get_setup_requires,
)

# read description
with open('README.md', 'rb') as f:
    longdesc = f.read().decode().strip()

# -- dependencies -----------

# build dependencies (dynamic based on arguments)
setup_requires = get_setup_requires()

# runtime dependencies
install_requires = [
    'astropy >= 3.0.0',
    'healpy',
    'matplotlib >= 3.1.0',
    'numpy >= 1.16.0',
    'scipy >= 1.2.0',
]

# test dependencies
tests_require = [
    "pytest >= 3.3.0",
    "pytest-cov >= 2.4.0",
]

# -- run setup ----------------------------------------------------------------

setup(
    # metadata
    name='pyschild',
    provides=['pyschild'],
    version=VERSION,
    description="A python package for gravitational-wave astrophysics",
    long_description=longdesc,
    long_description_content_type='text/markdown',
    author='Alex Urban',
    author_email='alexander.urban@ligo.org',
    license='GPL-3.0-or-later',
    url="https://pyschild.github.io",
    download_url="https://pyschild.github.io/docs/stable/install/",
    project_urls={
        "Bug Tracker": "https://github.com/pyschild/pyschild/issues",
        "Documentation": "https://pyschild.github.io/docs/",
        "Source Code": "https://github.com/pyschild/pyschild",
    },

    # package content
    packages=find_packages(),
    entry_points={},
    include_package_data=True,

    # dependencies
    cmdclass=CMDCLASS,
    python_requires=">=3.6",
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,

    # classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        ('License :: OSI Approved :: '
         'GNU General Public License v3 or later (GPLv3+)'),
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
