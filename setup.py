#from distutils.core import setup
#from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
import numpy

import setuptools

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension('cssm', ['src/cssm.pyx'], language = 'c++')], 
                                compiler_directives = {"language_level": "3"})
except ImportError:
    ext_modules = [Extension('cssm', ['src/cssm.pyx'], language = 'c++')]



setup(  
        name = 'SSMS',
        version='0.0.1',
        author = 'Alexander Fenger',
        url = 'https://github.com/AlexanderFengler/ssms',
        packages= ['ssms', 'ssms.basic_simulators', 'ssms.config', 'ssms.dataset_generators', 'ssms.support_utils'],
        description='SSMS is a package collecting simulators and training data generators for a bunch of generative models of interest in the cognitive science / neuroscience and approximate bayesian computation communities',
        install_requires= ['NumPy >= 1.17.0', 'SciPy >= 1.6.3', 'cython >= 0.29.23', 'pandas >= 1.2.4', 'scikit-learn >= 0.24.0'],
        setup_requires= ['NumPy >= 1.17.0', 'SciPy >= 1.6.3', 'cython >= 0.29.23', 'pandas >= 1.2.4', 'scikit-learn >= 0.24.0'],
        include_dirs = [numpy.get_include()] ,
        ext_modules = ext_modules ,
        classifiers=[ 'Development Status :: 1 - Planning', 
                      'Environment :: Console',
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Topic :: Scientific/Engineering'
                    ]

    )


# package_data={'hddm':['examples/*.csv', 'examples/*.conf', 'keras_models/*.h5', 'cnn_models/*/*', 'simulators/*']},
# scripts=['scripts/hddm_demo.py'],