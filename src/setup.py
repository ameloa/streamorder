from distutils.core import setup
from Cython.Build import cythonize

setup(name='Linear deterministic greedy',
      ext_modules=cythonize("alg.pyx"))
