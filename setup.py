# -*- coding: utf-8 -*-

import os
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

if __name__ == '__main__':
    from distutils.core import setup
    extra_files = package_files('EasyBeam')
    setup(name='EasyBeam',
          version='0.0.0',
          description='Easy Application for Structural analYsis with BEAMs',
          author='V. Gufler',
          url='https://github.com/veiguf/EasyBeam',
          package_data={'': extra_files},
          license='gpl-3.0',
          packages=['EasyBeam'],
          install_requires=['numpy',
                            'scipy',
                            'matplotlib'])
