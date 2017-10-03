from primitiv.device cimport _Device


cdef class _CPUDevice(_Device):

    def __cinit__(self, rng_seed = None):
        if rng_seed == None:
            self.wrapped = new CPUDevice()
        else:
            self.wrapped = new CPUDevice(<unsigned> rng_seed)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef CPUDevice *temp
        if self.wrapped is not NULL:
            temp = <CPUDevice*> self.wrapped
            del temp
            self.wrapped = NULL

    def dump_description(self):
        (<CPUDevice*> self.wrapped).dump_description()
        return

    def type(self):
        return (<CPUDevice*> self.wrapped).type()
