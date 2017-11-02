from primitiv._device cimport CppDevice, _Device


cdef extern from "primitiv/cuda_device.h":
    cdef cppclass CppCUDA "primitiv::devices::CUDA" (CppDevice):
        CppCUDA(unsigned device_id) except +
        CppCUDA(unsigned device_id, unsigned rng_seed) except +
        @staticmethod
        unsigned num_devices() except +


cdef class _CUDA(_Device):
    pass
