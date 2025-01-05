from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("c_MCTS.pyx"),
    include_dirs=[numpy.get_include()]
)

# setup(
#     ext_modules=cythonize("c_utils.pyx"),
#     include_dirs=[numpy.get_include()]
# )