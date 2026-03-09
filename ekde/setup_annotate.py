from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
     name="ekde.ekdefunc",
    ext_modules=cythonize(
        "cython/ekdefunc.pyx",
        annotate=True,
        ),
    include_dirs=[numpy.get_include()]
)    

