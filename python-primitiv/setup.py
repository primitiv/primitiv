#!/usr/bin/env python3

import sys
import numpy as np

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

enable_cuda = False
if "--enable-cuda" in sys.argv:
    enable_cuda = True
    sys.argv.remove("--enable-cuda")

def extension_common_args(*args, **kwargs):
    return Extension(*args, **kwargs,
        language="c++",
        libraries=["primitiv"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    )

ext_modules = [
    extension_common_args("primitiv._shape",
                          sources=["primitiv/_shape.pyx"]),
    extension_common_args("primitiv._tensor",
                          sources=["primitiv/_tensor.pyx"]),
    extension_common_args("primitiv._device",
                          sources=["primitiv/_device.pyx"]),
    extension_common_args("primitiv.devices._naive_device",
                          sources=["primitiv/devices/_naive_device.pyx"]),
    extension_common_args("primitiv._parameter",
                          sources=["primitiv/_parameter.pyx"]),
    extension_common_args("primitiv._initializer",
                          sources=["primitiv/_initializer.pyx"]),
    extension_common_args("primitiv.initializers._initializer_impl",
                          sources=["primitiv/initializers/_initializer_impl.pyx"]),
    extension_common_args("primitiv._graph",
                          sources=["primitiv/_graph.pyx"]),
    extension_common_args("primitiv._trainer",
                          sources=["primitiv/_trainer.pyx"]),
    extension_common_args("primitiv.trainers._trainer_impl",
                          sources=["primitiv/trainers/_trainer_impl.pyx"]),
    extension_common_args("primitiv._operator",
                          sources=["primitiv/_operator.pyx"]),
    extension_common_args("primitiv.config",
                          sources=["primitiv/config.pyx"]),
]

if enable_cuda:
    ext_modules.append(extension_common_args("primitiv.devices._cuda_device",
                                             sources=["primitiv/devices/_cuda_device.pyx"]))

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
    install_requires=[
        "cython",
        "numpy",
    ],
)
