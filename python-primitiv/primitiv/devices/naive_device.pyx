from primitiv.device cimport _Device


cdef class _Naive(_Device):

    def __init__(self, rng_seed = None):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if rng_seed == None:
            self.wrapped_newed = new Naive()
        else:
            self.wrapped_newed = new Naive(<unsigned> rng_seed)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef Naive *temp
        if self.wrapped_newed is not NULL:
            temp = <Naive*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    def dump_description(self):
        (<Naive*> self.wrapped).dump_description()
        return

    def type(self):
        return (<Naive*> self.wrapped).type()
