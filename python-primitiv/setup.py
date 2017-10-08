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
    Extension("primitiv.shape",
              sources=["primitiv/shape.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.tensor",
              sources=["primitiv/tensor.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.device",
              sources=["primitiv/device.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.devices.naive_device",
              sources=["primitiv/devices/naive_device.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.parameter",
              sources=["primitiv/parameter.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.initializer",
              sources=["primitiv/initializer.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.initializers.initializer_impl",
              sources=["primitiv/initializers/initializer_impl.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.graph",
              sources=["primitiv/graph.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.trainer",
              sources=["primitiv/trainer.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.trainers.trainer_impl",
              sources=["primitiv/trainers/trainer_impl.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.operator",
              sources=["primitiv/operator.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
    Extension("primitiv.default_scope",
              sources=["primitiv/default_scope.pyx"],
              language="c++",
              libraries=["primitiv"],
              extra_compile_args=["-std=c++11"],
    ),
]

if enable_cuda:
    ext_modules.append(
        Extension("primitiv.devices.cuda_device",
                  sources=["primitiv/devices/cuda_device.pyx"],
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
