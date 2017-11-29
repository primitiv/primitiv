from primitiv._device import Device
from primitiv._graph import Graph
from primitiv._initializer import Initializer
from primitiv._model import Model
from primitiv._graph import Node
from primitiv._parameter import Parameter
from primitiv._shape import Shape
from primitiv._tensor import Tensor
from primitiv._optimizer import Optimizer

from primitiv import devices
from primitiv import initializers
from primitiv._operator import operators
from primitiv._operator import tensor_operators
from primitiv import optimizers
from primitiv import config

# NOTE(vbkaisetsu):
# Python uses unicode for string management, but C++ only uses raw byte arrays.
# This code sets the current locale information for the default encoding to convert
# strings between Python and C++.
config.set_encoding()


__all__ = [
    "Device",
    "Graph",
    "Initializer",
    "Model",
    "Node",
    "Parameter",
    "Shape",
    "Tensor",
    "Optimizer",

    "devices",
    "initializers",
    "operators",
    "tensor_operators",
    "optimizers",
    "config",
]
