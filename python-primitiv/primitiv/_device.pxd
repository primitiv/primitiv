from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape


cdef extern from "primitiv/device.h" namespace "primitiv":
    cdef cppclass CppDevice "primitiv::Device":
        void dump_description() except +


cdef extern from "primitiv/device.h" namespace "primitiv::Device":
    cdef CppDevice &get_default()
    cdef void set_default(CppDevice &dev)


cdef class _Device:
    cdef CppDevice *wrapped
    cdef CppDevice *wrapped_newed


cdef inline _Device wrapDevice(CppDevice *wrapped) except +:
    cdef _Device device = _Device.__new__(_Device)
    device.wrapped = wrapped
    return device


cdef _Device py_primitiv_Device_default
