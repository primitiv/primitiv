from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport _Device
from primitiv._tensor cimport _Tensor
from primitiv._shape cimport wrapShape, normShape
from utils cimport ndarrays_to_vector

from weakref import WeakValueDictionary

import numpy as np

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_parameter_weak_dict = WeakValueDictionary()


cdef class _ParameterStatistics:

    def __init__(self, _Parameter param):
        self.param = param

    def __getitem__(self, str name):
        return _Tensor.get_wrapper(&self.param.wrapped.stats(<string> name.encode("utf-8")))

    def __setitem__(self, str name, _Tensor value):
        cdef CppTensor *tensor_p = &self.param.wrapped.stats(<string> name.encode("utf-8"))
        if tensor_p == value.wrapped:
            return
        tensor_p[0] = value.wrapped[0]

    def __contains__(self, str name):
        return self.param.wrapped.has_stats(name.encode("utf-8"))


cdef class _Parameter:

    def __cinit__(self):
        self.stats = _ParameterStatistics(self)

    def __init__(self, str name, shape = None, init = None, _Device device = None):
        if self.wrapped is not NULL:
            raise MemoryError()
        if device is None:
            device = _Device.get_default()
        # Parameter(name, shape, np.ndarray init, device) new from ndarray
        if isinstance(init, np.ndarray):
            if shape is None:
                shape = _Shape(init.shape, 1)
            self.wrapped = new CppParameter(<string> name.encode("utf-8"), normShape(shape).wrapped, ndarrays_to_vector([init]), device.wrapped[0])
        # Parameter(name, shape, device)
        elif shape is not None and init is None:
            self.wrapped = new CppParameter(<string> name.encode("utf-8"), normShape(shape).wrapped, device.wrapped[0])
        # Parameter(name, shape, Initializer init, device) new from Initializer
        elif shape is not None and isinstance(init, _Initializer):
            self.wrapped = new CppParameter(<string> name.encode("utf-8"), normShape(shape).wrapped, (<_Initializer> init).wrapped[0], device.wrapped[0])
        elif isinstance(init, list):
            # Parameter(name, shape, vector<float> init, device) new from float list
            if shape is None:
                raise TypeError("shape is required when init is a list")
            self.wrapped = new CppParameter(<string> name.encode("utf-8"), normShape(shape).wrapped, <vector[float]> init, device.wrapped[0])
        else:
            raise TypeError("Argument 'init' has incorrect type (list, Initializer, or numpy.ndarray)")
        if self.wrapped is NULL:
            raise MemoryError()
        _Parameter.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    def valid(self):
        return self.wrapped.valid()

    def reset_value_by_vector(self, vector[float] &value):
        self.wrapped.reset_value(value)
        return

    def reset_value_by_initializer(self, _Initializer init):
        self.wrapped.reset_value(init.wrapped[0])
        return

    def reset_gradient(self):
        self.wrapped.reset_gradient()
        return

    def add_stats(self, str name, shape):
        self.wrapped.add_stats(<string> name.encode("utf-8"), normShape(shape).wrapped)
        return

    # NOTE(vbkaisetsu)
    # `has_stats` function is removed in Python.
    # Use "in" operator of `stats` variable instead.
    # def has_stats(self, str name):
    #     return self.wrapped.has_stats(name.encode("utf-8"))

    def name(self):
        return self.wrapped.name().decode("utf-8")

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return _Device.get_wrapper(&self.wrapped.device())

    # NOTE(vbkaisetsu)
    # `value` function is replaced with a property in Python.
    @property
    def value(self):
        return _Tensor.get_wrapper(&self.wrapped.value())

    @value.setter
    def value(self, _Tensor value):
        cdef CppTensor *tensor_p = &self.wrapped.value()
        if tensor_p == value.wrapped:
            return
        tensor_p[0] = value.wrapped[0]

    def gradient(self):
        return _Tensor.get_wrapper(&self.wrapped.gradient())

    # NOTE(vbkaisetsu)
    # This function is replaced with `stats` variable.
    # def stats(self, str name):
    #     return _Tensor.get_wrapper(&self.wrapped.stats(<string> name.encode("utf-8")))

    def save(self, str path, bool with_stats = True):
        self.wrapped.save(<string> path.encode("utf-8"), with_stats)
        return

    @staticmethod
    def load(str path, bool with_stats = True, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Parameter.get_wrapper_with_new(Parameter_load(<string> path.encode("utf-8"), with_stats, device.wrapped[0]))

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppParameter *ptr, _Parameter wrapper):
        if <uintptr_t> ptr in py_primitiv_parameter_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_parameter_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef _Parameter get_wrapper(CppParameter *ptr):
        return py_primitiv_parameter_weak_dict[<uintptr_t> ptr]

    @staticmethod
    cdef _Parameter get_wrapper_with_new(CppParameter *ptr):
        cdef _Parameter param = _Parameter.__new__(_Parameter)
        param.wrapped = ptr
        if py_primitiv_parameter_weak_dict.setdefault(<uintptr_t> ptr, param) is not param:
            raise ValueError("Attempted to register the same C++ object twice.")
        return param
