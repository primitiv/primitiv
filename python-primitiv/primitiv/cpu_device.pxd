from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cpu_device.h" namespace "primitiv":
    cdef cppclass CPUDevice(Device):
        CPUDevice() except +
        CPUDevice(unsigned rng_seed) except +
        void dump_description() except +
        Device.DeviceType type() except +


cdef class _CPUDevice(_Device):
    pass


cdef inline _CPUDevice wrapCPUDevice(CPUDevice *wrapped) except +:
    cdef _CPUDevice cpu_device = _CPUDevice.__new__(_CPUDevice)
    cpu_device.wrapped = wrapped
    return cpu_device
