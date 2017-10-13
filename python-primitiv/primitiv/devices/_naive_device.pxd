from primitiv._device cimport CppDevice, _Device


cdef extern from "primitiv/naive_device.h" namespace "primitiv::devices":
    cdef cppclass CppNaive "primitiv::devices::Naive" (CppDevice):
        CppNaive() except +
        CppNaive(unsigned rng_seed) except +


cdef class _Naive(_Device):
    pass


cdef inline _Naive wrapNaive(CppNaive *wrapped) except +:
    cdef _Naive naive = _Naive.__new__(_Naive)
    naive.wrapped = wrapped
    return naive
