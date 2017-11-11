from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport _Device
from primitiv._tensor cimport _Tensor
from primitiv._shape cimport wrapShape, normShape
from utils cimport ndarrays_to_vector
from primitiv.config cimport pystr_to_cppstr

from weakref import WeakValueDictionary

import numpy as np
import weakref

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_parameter_weak_dict = WeakValueDictionary()


cdef class _ParameterStatistics:

    def __init__(self, _Parameter param):
        # NOTE(vbkaisetsu): It becomes circular reference.
        # We can't know when it will be deleted by the garbage collector.
        # Therefore we hold this instance in a weakref to delete it immediately.
        self.param_ref = weakref.ref(param)

    def __getitem__(self, str name):
        return _Tensor.get_wrapper(&(<_Parameter> self.param_ref()).wrapped.stats(pystr_to_cppstr(name)))

    def __setitem__(self, str name, _Tensor value):
        cdef CppTensor *tensor_p = &(<_Parameter> self.param_ref()).wrapped.stats(pystr_to_cppstr(name))
        tensor_p[0] = value.wrapped[0]

    def __contains__(self, str name):
        return (<_Parameter> self.param_ref()).wrapped.has_stats(pystr_to_cppstr(name))


cdef class _Parameter:

    def __cinit__(self):
        self.stats = _ParameterStatistics(self)

    def __init__(self, shape = None, initializer = None, _Device device = None):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppParameter()
        self.init(shape, initializer, device)
        _Parameter.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    def init(self, shape = None, initializer = None, _Device device = None):
        if device is None:
            device = _Device.get_default()
        if isinstance(initializer, np.ndarray):
            if shape is None:
                shape = _Shape(initializer.shape, 1)
            self.wrapped.init(normShape(shape).wrapped, ndarrays_to_vector([initializer]), device.wrapped[0])
        elif isinstance(initializer, _Initializer):
            if shape is None:
                raise TypeError("shape is required when initializer is an Initializer")
            self.wrapped.init(normShape(shape).wrapped, (<_Initializer> initializer).wrapped[0], device.wrapped[0])
        elif isinstance(initializer, list):
            if shape is None:
                raise TypeError("shape is required when initializer is a list")
            self.wrapped.init(normShape(shape).wrapped, <vector[float]> initializer, device.wrapped[0])
        elif initializer is None:
            if shape is not None:
                raise TypeError("shape is given but initializer is not given")
        else:
            raise TypeError("Argument 'initializer' has incorrect type (list, Initializer, or numpy.ndarray)")
        return

    def load(self, str path, bool with_stats = True, _Device device = None):
        if device is None:
            device = _Device.get_default()
        self.wrapped.load(pystr_to_cppstr(path), with_stats, device.wrapped[0])
        return

    def save(self, str path, bool with_stats = True):
        self.wrapped.save(pystr_to_cppstr(path), with_stats)
        return

    def valid(self):
        return self.wrapped.valid()

    def reset_gradient(self):
        self.wrapped.reset_gradient()
        return

    def add_stats(self, str name, shape):
        self.wrapped.add_stats(pystr_to_cppstr(name), normShape(shape).wrapped)
        return

    # NOTE(vbkaisetsu):
    # `has_stats` function is removed in Python.
    # Use "in" operator of `stats` variable instead.
    # def has_stats(self, str name):
    #     return self.wrapped.has_stats(pystr_to_cppstr(name))

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return _Device.get_wrapper(&self.wrapped.device())

    # NOTE(vbkaisetsu):
    # `value` function is replaced with a property in Python.
    @property
    def value(self):
        return _Tensor.get_wrapper(&self.wrapped.value())

    @value.setter
    def value(self, _Tensor value):
        cdef CppTensor *tensor_p = &self.wrapped.value()
        tensor_p[0] = value.wrapped[0]

    # NOTE(vbkaisetsu):
    # `gradient` function is replaced with a property in Python.
    @property
    def gradient(self):
        return _Tensor.get_wrapper(&self.wrapped.gradient())

    @gradient.setter
    def gradient(self, _Tensor value):
        cdef CppTensor *tensor_p = &self.wrapped.gradient()
        tensor_p[0] = value.wrapped[0]

    # NOTE(vbkaisetsu):
    # This function is replaced with `stats` variable.
    # def stats(self, str name):
    #     return _Tensor.get_wrapper(&self.wrapped.stats(pystr_to_cppstr(name)))

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
