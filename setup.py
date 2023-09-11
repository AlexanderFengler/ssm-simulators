#from distutils.core import setup
from xml.etree.ElementInclude import include
from setuptools import setup, Extension
import numpy

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension('cssm', ['src/cssm.pyx'], language = 'c++')], 
                                compiler_directives = {"language_level": "3"})
except ImportError:
    ext_modules = [Extension('cssm', ['src/cssm.pyx'], language = 'c++')]

setup(
    packages=['ssms', 'ssms.basic_simulators', 'ssms.config', 'ssms.dataset_generators', 'ssms.support_utils'],
    include_dirs = [numpy.get_include()],
    ext_modules = ext_modules,
)

# setup(  
#         name = 'ssm-simulators',
#         version='0.3.1',
#         author = 'Alexander Fenger',
#         url = 'https://github.com/AlexanderFengler/ssms',
#         packages= ['ssms', 'ssms.basic_simulators', 'ssms.config', 'ssms.dataset_generators', 'ssms.support_utils'],
#         description='SSMS is a package collecting simulators and training data generators for a bunch of generative models of interest in the cognitive science / neuroscience and approximate bayesian computation communities',
#         install_requires= ['numpy >= 1.17.0', 'scipy >= 1.6.3', 'cython >= 0.29.23', 'pandas >= 1.0.0', 'scikit-learn >= 0.24.0', 'psutil >= 5.0.0'],
#         setup_requires= ['numpy >= 1.17.0', 'scipy >= 1.6.3', 'cython >= 0.29.23', 'pandas >= 1.0.0', 'scikit-learn >= 0.24.0', 'psutil >= 5.0.0'],
#         include_dirs = [numpy.get_include()] ,
#         ext_modules = ext_modules ,
#         classifiers=[ 'Development Status :: 1 - Planning', 
#                       'Environment :: Console',
#                       'License :: OSI Approved :: MIT License',
#                       'Programming Language :: Python',
#                       'Topic :: Scientific/Engineering'
#                     ]
#     )