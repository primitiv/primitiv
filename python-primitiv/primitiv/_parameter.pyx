from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport Device
from primitiv._tensor cimport Tensor
from primitiv._shape cimport wrapShape, normShape
from utils cimport ndarrays_to_vector
from primitiv.config cimport pystr_to_cppstr

from weakref import WeakValueDictionary

import numpy as np
cimport numpy as np
import weakref

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_parameter_weak_dict = WeakValueDictionary()


cdef class ParameterStatistics:

    def __init__(self, Parameter param):
        # NOTE(vbkaisetsu): It becomes circular reference.
        # We can't know when it will be deleted by the garbage collector.
        # Therefore we hold this instance in a weakref to delete it immediately.
        self.param_ref = weakref.ref(param)

    def __getitem__(self, str name):
        return Tensor.get_wrapper(&(<Parameter> self.param_ref()).wrapped.stats(pystr_to_cppstr(name)))

    def __setitem__(self, str name, Tensor value):
        cdef CppTensor *tensor_p = &(<Parameter> self.param_ref()).wrapped.stats(pystr_to_cppstr(name))
        tensor_p[0] = value.wrapped[0]

    def __contains__(self, str name):
        return (<Parameter> self.param_ref()).wrapped.has_stats(pystr_to_cppstr(name))


cdef class Parameter:
    """Class to manage a trainable tensor parameter.

    """

    def __cinit__(self):
        self.stats = ParameterStatistics(self)

    def __init__(self, *args, **kwargs):
        """Creates a new Parameter object.

        If no argument is given, a Parameter is initialized with zeros,
        otherwise ``init()`` method is called with given arguments.

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppParameter()
        if len(args) != 0 or len(kwargs) != 0:
            self.init(*args, **kwargs)
        Parameter.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    # NOTE(vbkaisetsu):
    # Python's Parameter.init only takes shape+Initializer arguments.
    def init(self, shape, Initializer initializer, Device device = None):
        """Initializes the Parameter object.

        :param shape: The shape of the parameter. The batch size should be 1.
        :type shape: primitiv.Shape
        :param init: An Initializer object.
        :type init: primitiv.Initializer
        :param device: The device object to manage internal memory (default: ``None``).
        :type device: primitiv.Device or None

        """
        if device is None:
            device = Device.get_default()
        self.wrapped.init(normShape(shape).wrapped, initializer.wrapped[0], device.wrapped[0])

    def load(self, str path, bool with_stats = True, Device device = None):
        """Loads parameters from specified file.

        :param path: File path to load parameters.
        :type path: str
        :param with_stats: Whether or not to load all additional statistics as well
                           as parameter values if the file has them (default: True).
        :type with_stats: bool
        :param device: The device object to manage internal memory (default: None).
        :type device: primitiv.Device or None

        """
        if device is None:
            device = Device.get_default()
        self.wrapped.load(pystr_to_cppstr(path), with_stats, device.wrapped[0])
        return

    def save(self, str path, bool with_stats = True):
        """Saves current parameters into specified file.

        :param path: File path to save parameters.
        :type path: str
        :param with_stats: Whether or not to save all additional statistics as well
                           as parameter values if the parameter object has them
                           (default: ``True``).
        :type with_stats: bool

        """
        self.wrapped.save(pystr_to_cppstr(path), with_stats)
        return

    def valid(self):
        """Returns whether the parameter is valid or not.

        :return: ``True`` or ``False`` w.r.t. the parameter is valid or not.
        :rtype: bool

        """
        return self.wrapped.valid()

    def reset_gradient(self):
        """Set all gradients to 0.

        """
        self.wrapped.reset_gradient()
        return

    def add_stats(self, str name, shape):
        """Adds a new optional statistics tensor.

        :param name: Name of the statistics.
        :rtype: str
        :param shape: Shape of the tensor.
        :rtype: primitiv.Shape

        All elements in the new statistics tensor is initialized by 0.

        """
        self.wrapped.add_stats(pystr_to_cppstr(name), normShape(shape).wrapped)
        return

    # NOTE(vbkaisetsu):
    # `has_stats` function is removed in Python.
    # Use "in" operator of `stats` variable instead.
    # def has_stats(self, str name):
    #     return self.wrapped.has_stats(pystr_to_cppstr(name))

    def shape(self):
        """Returns the shape of the parameter.

        :return: Shape of the parameter.
        :rtype: primitiv.Shape

        """
        return wrapShape(self.wrapped.shape())

    def device(self):
        """Returns the Device object to manage the internal memory.

        :return: The Device object.
        :rtype: primitiv.Device

        """
        return Device.get_wrapper(&self.wrapped.device())

    # NOTE(vbkaisetsu):
    # `value` function is replaced with a property in Python.
    @property
    def value(self):
        """A ``Tensor`` representing the parameter tensor.

        """
        return Tensor.get_wrapper(&self.wrapped.value())

    @value.setter
    def value(self, Tensor value):
        cdef CppTensor *tensor_p = &self.wrapped.value()
        tensor_p[0] = value.wrapped[0]

    # NOTE(vbkaisetsu):
    # `gradient` function is replaced with a property in Python.
    @property
    def gradient(self):
        """A ``Tensor`` representing the current gradient of the value.

        """
        return Tensor.get_wrapper(&self.wrapped.gradient())

    @gradient.setter
    def gradient(self, Tensor value):
        cdef CppTensor *tensor_p = &self.wrapped.gradient()
        tensor_p[0] = value.wrapped[0]

    # NOTE(vbkaisetsu):
    # This function is replaced with `stats` variable.
    # def stats(self, str name):
    #     return Tensor.get_wrapper(&self.wrapped.stats(pystr_to_cppstr(name)))

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppParameter *ptr, Parameter wrapper):
        if <uintptr_t> ptr in py_primitiv_parameter_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_parameter_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef Parameter get_wrapper(CppParameter *ptr):
        return py_primitiv_parameter_weak_dict[<uintptr_t> ptr]

    @staticmethod
    cdef Parameter get_wrapper_with_new(CppParameter *ptr):
        cdef Parameter param = Parameter.__new__(Parameter)
        param.wrapped = ptr
        if py_primitiv_parameter_weak_dict.setdefault(<uintptr_t> ptr, param) is not param:
            raise ValueError("Attempted to register the same C++ object twice.")
        return param
