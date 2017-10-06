from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cuda_device.h" namespace "primitiv::devices":
    cdef cppclass CUDA(Device):
        CUDA(unsigned device_id) except +
        CUDA(unsigned device_id, unsigned rng_seed) except +
        void dump_description() except +
        Device.DeviceType type() except +


cdef extern from "primitiv/cuda_device.h" namespace "primitiv::devices::CUDA":
    cdef unsigned num_devices() except +


cdef class _CUDA(_Device):
    pass


cdef inline _CUDA wrapCUDA(CUDA *wrapped) except +:
    cdef _CUDA cuda = _CUDA.__new__(_CUDA)
    cuda.wrapped = wrapped
    return cuda
