# from distutils.core import setup
# from xml.etree.ElementInclude import include
from setuptools import setup, Extension
import numpy

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [Extension("cssm", ["src/cssm.pyx"], language="c++")],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = [Extension("cssm", ["src/cssm.pyx"], language="c++")]

setup(
    packages=[
        "ssms",
        "ssms.basic_simulators",
        "ssms.config",
        "ssms.dataset_generators",
        "ssms.support_utils",
    ],
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
)
