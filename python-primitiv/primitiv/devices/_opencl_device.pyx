from primitiv._device cimport _Device


cdef class _OpenCL(_Device):

    def __init__(self, unsigned platform_id, unsigned device_id):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        self.wrapped = new CppOpenCL(platform_id, device_id)
        _Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL
