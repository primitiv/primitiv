from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

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
    cdef object __weakref__


cdef object py_primitiv_device_weak_dict

cdef inline _Device wrapDevice(CppDevice *wrapped) except +:
    global py_primitiv_device_weak_dict
    if (<uintptr_t> wrapped) in py_primitiv_device_weak_dict:
        return py_primitiv_device_weak_dict[<uintptr_t> wrapped]
    cdef _Device device = _Device.__new__(_Device)
    device.wrapped = wrapped
    return device
