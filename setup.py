#!/usr/bin/env python
# This file is part of PSAMM.
#
# PSAMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PSAMM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PSAMM.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015-2016  Jon Lund Steffensen <jon_steffensen@uri.edu>

from setuptools import setup, find_packages

setup(
    name='psamm-import-matlab',
    version='0.3',
    description='PSAMM importer for Matlab (.mat) COBRA models',
    maintainer='Jon Lund Steffensen',
    maintainer_email='jon_steffensen@uri.edu',
    url='https://github.com/zhanglab/psamm-import-matlab',
    license='GNU GPLv3+',

    classifiers = [
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
    ],

    packages=find_packages(),
    entry_points = {
        'psamm.importer': [
            'matlab = psamm_import_matlab:Importer',
        ]
    },

    install_requires=[
        'psamm>=0.19',
        'psamm-import>=0.10',
        'numpy',
        'scipy',
        'six'
    ])
