from primitiv._device cimport CppDevice, _Device


cdef extern from "primitiv/opencl_device.h":
    cdef cppclass CppOpenCL "primitiv::devices::OpenCL" (CppDevice):
        CppOpenCL(unsigned platform_id, unsigned device_id) except +
        CppOpenCL(unsigned platform_id, unsigned device_id, unsigned rng_seed) except +
        @staticmethod
        unsigned num_platforms() except +
        @staticmethod
        unsigned num_devices(unsigned platform_id) except +


cdef class _OpenCL(_Device):
    pass
