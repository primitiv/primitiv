from primitiv._device cimport _Device


cdef class _Naive(_Device):

    def __init__(self, rng_seed = None):
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if rng_seed is None:
            self.wrapped = new CppNaive()
        else:
            self.wrapped = new CppNaive(<unsigned> rng_seed)

        _Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL
