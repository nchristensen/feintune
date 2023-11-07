#!/usr/bin/env python

# run pipreqs to generate requirements.txt

from distutils.core import setup

setup(name='Feintune',
      version='2023.0.0',
      description='Autotuner for batched einsum Loopy kernels',
      # author='',
      # author_email='',
      # url='',
      packages=['feintune'],
      install_requires=[
          'ConfigSpace',
          'dh_scikit_optimize',
          'autotune',
          'immutabledict',
          'ytopt',
          'func_timeout',
          'hjson',
          'islpy',
          'matplotlib',
          'mpi4py',
          'numpy',
          'pandas',
          'pyopencl',
          'pytools',
          # 'beautifulsoup4',
          # 'charm4py',
          # 'guppy',
          # 'mem_top',
          # 'mpipool',
          'Pebble',
          'psutil',
          'pyinstrument',
          # 'schwimmbad',
          # 'grudge',
          'meshmode',
          # 'mirgecom',
          'loopy'
      ]
      )
