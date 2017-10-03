from primitiv.device cimport _Device
from primitiv.cuda_device cimport num_devices as CUDADevice_num_devices


cdef class _CUDADevice(_Device):

    def __cinit__(self, rng_seed = None):
        if rng_seed == None:
            self.wrapped = new CUDADevice()
        else:
            self.wrapped = new CUDADevice(<unsigned> rng_seed)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef CUDADevice *temp
        if self.wrapped is not NULL:
            temp = <CUDADevice*> self.wrapped
            del temp
            self.wrapped = NULL

    @staticmethod
    def num_devices():
        return CUDADevice_num_devices()

    def dump_description(self):
        (<CUDADevice*> self.wrapped).dump_description()
        return

    def type(self):
        return (<CUDADevice*> self.wrapped).type()
