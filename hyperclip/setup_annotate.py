from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
     name="hyperclip.hyperfunc",
    ext_modules=cythonize(
        "cython/hyperfunc.pyx",
        annotate=True,
        ),
    include_dirs=[numpy.get_include()]
)    

