from primitiv.device cimport _Device
from primitiv.cuda_device cimport num_devices as CUDA_num_devices


cdef class _CUDA(_Device):

    def __cinit__(self, rng_seed = None):
        if rng_seed == None:
            self.wrapped = new CUDA()
        else:
            self.wrapped = new CUDA(<unsigned> rng_seed)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef CUDA *temp
        if self.wrapped is not NULL:
            temp = <CUDA*> self.wrapped
            del temp
            self.wrapped = NULL

    @staticmethod
    def num_devices():
        return CUDA_num_devices()

    def dump_description(self):
        (<CUDADevice*> self.wrapped).dump_description()
        return

    def type(self):
        return (<CUDADevice*> self.wrapped).type()
