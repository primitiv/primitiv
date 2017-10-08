from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport _Device, wrapDevice
from primitiv.tensor cimport wrapTensor
from primitiv.shape cimport wrapShape, normShape
from primitiv.default_scope cimport _DefaultScopeDevice

import numpy as np

from utils cimport ndarrays_to_vector


cdef class _Parameter:

    def __init__(self, str name, shape = None, init = None, _Device device = None):
        if self.wrapped_newed is not NULL:
            raise MemoryError()

        if device == None:
            device = _DefaultScopeDevice.get()

        # Parameter(name, shape, np.ndarray init, device) new from ndarray
        if isinstance(init, np.ndarray):
            if shape is None:
                shape = _Shape(init.shape, 1)
            self.wrapped_newed = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, ndarrays_to_vector([init]), device.wrapped[0])

        # Parameter(name, shape, device)
        elif shape is not None and init is None:
            self.wrapped_newed = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, device.wrapped[0])

        # Parameter(name, shape, Initializer init, device) new from Initializer
        elif shape is not None and isinstance(init, _Initializer):
            self.wrapped_newed = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, (<_Initializer> init).wrapped[0], device.wrapped[0])

        elif isinstance(init, list):
            # Parameter(name, shape, vector<float> init, device) new from float list
            if shape is None:
                raise TypeError("shape is required when init is a list")
            self.wrapped_newed = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, <vector[float]> init, device.wrapped[0])

        else:
            raise TypeError("Argument 'init' has incorrect type (list, Initializer, or numpy.ndarray)")

        if self.wrapped_newed is NULL:
            raise MemoryError()

        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
            self.wrapped_newed = NULL

    #def copy(self):
        #return wrapParameter(new Parameter(self.wrapped[0]))

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

    def has_stats(self, str name):
        return self.wrapped.has_stats(name)

    def name(self):
        return self.wrapped.name().decode("utf-8")

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

    def value(self):
        return wrapTensor(self.wrapped.value())

    def gradient(self):
        return wrapTensor(self.wrapped.gradient())

    def stats(self, str name):
        return wrapTensor(self.wrapped.stats(<string> name.encode("utf-8")))

    def save(self, str path, bool with_stats = True):
        self.wrapped.save(<string> path.encode("utf-8"), with_stats)
        return

    @staticmethod
    def load(str path, bool with_stats = True, _Device device = None):
        if device == None:
            device = _DefaultScopeDevice.get()
        return wrapParameterWithNew(Parameter_load(<string> path.encode("utf-8"), with_stats, device.wrapped[0]))
