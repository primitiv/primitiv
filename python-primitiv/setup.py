#!/usr/bin/env python3

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("primitiv.shape",
              sources=["primitiv/shape.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.tensor",
              sources=["primitiv/tensor.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.device",
              sources=["primitiv/device.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.devices.naive_device",
              sources=["primitiv/devices/naive_device.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    #Extension("primitiv.devices.cuda_device",
    #sources=["primitiv/devices/cuda_device.pyx"],
              #language="c++",
              #libraries=["primitiv"]
    #),
    Extension("primitiv.function",
              sources=["primitiv/function.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.functions.function_impl",
              sources=["primitiv/functions/function_impl.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.parameter",
              sources=["primitiv/parameter.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.initializer",
              sources=["primitiv/initializer.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.initializers.initializer_impl",
              sources=["primitiv/initializers/initializer_impl.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.graph",
              sources=["primitiv/graph.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.trainer",
              sources=["primitiv/trainer.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.trainers.trainer_impl",
              sources=["primitiv/trainers/trainer_impl.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.operator",
              sources=["primitiv/operator.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.default_scope",
              sources=["primitiv/default_scope.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
]

setup(
    ext_modules = cythonize(ext_modules),
    packages = ["primitiv",
                "primitiv.devices",
                "primitiv.functions",
                "primitiv.initializers",
                "primitiv.trainers",
    ],
)
