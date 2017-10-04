from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport _Device, wrapDevice
from primitiv.tensor cimport wrapTensor
from primitiv.shape cimport wrapShape, normShape
from primitiv.default_scope cimport _DefaultScopeDevice

import numpy as np

from utils cimport ndarrays_to_vector


cdef class _Parameter:

    @staticmethod
    def new_from_list(str name, _Shape shape, vector[float] init, _Device device = None):
        cdef Parameter *param
        if device == None:
            device = _DefaultScopeDevice.get()
        param = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, init, device.wrapped[0])
        if param is NULL:
            raise MemoryError()
        return wrapParameter(param)

    @staticmethod
    def new_from_ndarrays(str name, list init, _Device device = None):
        cdef Parameter *param
        if device == None:
            device = _DefaultScopeDevice.get()
        if len(init) == 0:
            raise TypeError("init contains no item")
        shape = _Shape(init[0].shape, len(init))
        param = new Parameter(<string> name.encode("utf-8"), shape.wrapped, ndarrays_to_vector(init), device.wrapped[0])
        if param is NULL:
            raise MemoryError()
        return wrapParameter(param)

    @staticmethod
    def new_from_initializer(str name, shape, _Initializer init, _Device device = None):
        cdef Parameter *param
        if device == None:
            device = _DefaultScopeDevice.get()
        param = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, init.wrapped[0], device.wrapped[0])
        if param is NULL:
            raise MemoryError()
        return wrapParameter(param)

    def __init__(self, str name, shape, _Device device = None):
        if device == None:
            device = _DefaultScopeDevice.get()
        self.wrapped = new Parameter(<string> name.encode("utf-8"), normShape(shape).wrapped, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

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
        return wrapParameter(Parameter_load(<string> path.encode("utf-8"), with_stats, device.wrapped[0]))
