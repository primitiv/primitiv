#!/usr/bin/env python3

import sys

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

enable_cuda = False
if "--enable-cuda" in sys.argv:
    enable_cuda = True
    sys.argv.remove("--enable-cuda")

ext_modules = [
    Extension("primitiv._shape",
              sources=["primitiv/_shape.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._tensor",
              sources=["primitiv/_tensor.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._device",
              sources=["primitiv/_device.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.devices._naive_device",
              sources=["primitiv/devices/_naive_device.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._parameter",
              sources=["primitiv/_parameter.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._initializer",
              sources=["primitiv/_initializer.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.initializers._initializer_impl",
              sources=["primitiv/initializers/_initializer_impl.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._graph",
              sources=["primitiv/_graph.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._trainer",
              sources=["primitiv/_trainer.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.trainers._trainer_impl",
              sources=["primitiv/trainers/_trainer_impl.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv._operator",
              sources=["primitiv/_operator.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
]

if enable_cuda:
    ext_modules.append(
        Extension("primitiv.devices._cuda_device",
                  sources=["primitiv/devices/_cuda_device.pyx"],
                  language="c++",
                  libraries=["primitiv"],
                  extra_compile_args=["-std=c++11"],
        )
    )

setup(
    name = "primitiv",
    version = "0.0.1",
    description = "primitiv: A Neural Network Toolkit. (Python frontend)",
    url = "https://github.com/odashi/primitiv",
    author = "Koichi Akabe",
    author_email = "vbkaisetsu at gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules = cythonize(ext_modules),
    packages = [
        "primitiv",
        "primitiv.devices",
        "primitiv.initializers",
        "primitiv.trainers",
    ],
)
