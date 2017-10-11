from primitiv._device cimport Device, _Device


cdef extern from "primitiv/naive_device.h" namespace "primitiv::devices":
    cdef cppclass Naive(Device):
        Naive() except +
        Naive(unsigned rng_seed) except +
        void dump_description() except +
        Device.DeviceType type() except +


cdef class _Naive(_Device):
    pass


cdef inline _Naive wrapNaive(Naive *wrapped) except +:
    cdef _Naive naive = _Naive.__new__(_Naive)
    naive.wrapped = wrapped
    return naive
