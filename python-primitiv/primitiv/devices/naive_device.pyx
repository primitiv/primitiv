from primitiv.device cimport _Device


cdef class _Naive(_Device):

    def __cinit__(self, rng_seed = None):
        if rng_seed == None:
            self.wrapped = new Naive()
        else:
            self.wrapped = new Naive(<unsigned> rng_seed)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Naive *temp
        if self.wrapped is not NULL:
            temp = <Naive*> self.wrapped
            del temp
            self.wrapped = NULL

    def dump_description(self):
        (<Naive*> self.wrapped).dump_description()
        return

    def type(self):
        return (<Naive*> self.wrapped).type()
