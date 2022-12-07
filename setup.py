# -*- coding: utf-8 -*-

import os
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

if __name__ == '__main__':
    from setuptools import setup
    extra_files = package_files('EasyBeam')
    setup(
        name='EasyBeam',
        version='1.1.2',
        description='Easy Application for Structural analYsis with BEAMs',
        author='V. Gufler, E. J. Wehrle',
        author_email='veit.gufler@unibz.it',
        url='https://github.com/veiguf/EasyBeam',
        package_data={'': extra_files},
        license='GNU Lesser General Public License 3.0',
        packages=['EasyBeam'],
        copyright='Copyright 2020-2022 V. Gufler',
        install_requires=['numpy',
                          'scipy',
                          'matplotlib'],
        long_description=long_description,
        long_description_content_type='text/markdown',
    )
