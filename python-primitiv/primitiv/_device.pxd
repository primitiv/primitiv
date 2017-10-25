from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape


cdef extern from "primitiv/device.h":
    cdef cppclass CppDevice "primitiv::Device":
        void dump_description() except +


cdef extern from "primitiv/device.h":
    cdef CppDevice &CppDevice_get_default "primitiv::Device::get_default"()
    cdef void CppDevice_set_default "primitiv::Device::set_default"(CppDevice &dev)


cdef class _Device:
    cdef CppDevice *wrapped
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, _Device wrapper)
    @staticmethod
    cdef _Device get_wrapper(CppDevice *ptr)
