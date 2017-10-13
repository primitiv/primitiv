from primitiv._device cimport _Device


cdef class _Naive(_Device):

    def __init__(self, rng_seed = None):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if rng_seed == None:
            self.wrapped_newed = new CppNaive()
        else:
            self.wrapped_newed = new CppNaive(<unsigned> rng_seed)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CppNaive *temp
        if self.wrapped_newed is not NULL:
            temp = <CppNaive*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL
