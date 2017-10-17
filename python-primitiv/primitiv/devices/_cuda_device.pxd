from primitiv._device cimport CppDevice, _Device


cdef extern from "primitiv/cuda_device.h" namespace "primitiv::devices":
    cdef cppclass CppCUDA "primitiv::devices::CUDA" (CppDevice):
        CppCUDA(unsigned device_id) except +
        CppCUDA(unsigned device_id, unsigned rng_seed) except +


cdef extern from "primitiv/cuda_device.h" namespace "primitiv::devices::CUDA":
    cdef unsigned num_devices() except +


cdef class _CUDA(_Device):
    pass


cdef inline _CUDA wrapCUDA(CppCUDA *wrapped) except +:
    cdef _CUDA cuda = _CUDA.__new__(_CUDA)
    cuda.wrapped = wrapped
    return cuda
