from primitiv._device cimport _Device


cdef class _CUDA(_Device):

    def __init__(self, unsigned device_id, rng_seed = None):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if rng_seed is None:
            self.wrapped = new CppCUDA(device_id)
        else:
            self.wrapped = new CppCUDA(device_id, <unsigned> rng_seed)
        _Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def num_devices():
        return CppCUDA.num_devices()
