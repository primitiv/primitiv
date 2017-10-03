from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cuda_device.h" namespace "primitiv":
    cdef cppclass CUDADevice(Device):
        CUDADevice(unsigned device_id) except +
        CUDADevice(unsigned device_id, unsigned rng_seed) except +
        void dump_description() except +
        Device.DeviceType type() except +


cdef extern from "primitiv/cuda_device.h" namespace "primitiv::CPUDevice":
    cdef unsigned num_devices() except +


cdef class _CPUDevice(_Device):
    pass


cdef inline _CUDADevice wrapCUDADevice(CUDADevice *wrapped) except +:
    cdef _CUDADevice cuda_device = _CUDADevice.__new__(_CUDADevice)
    cuda_device.wrapped = wrapped
    return cuda_device
