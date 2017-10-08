from primitiv.device cimport _Device
from primitiv.devices.cuda_device cimport num_devices as CUDA_num_devices


cdef class _CUDA(_Device):

    def __init__(self, unsigned device_id, rng_seed = None):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if rng_seed == None:
            self.wrapped_newed = new CUDA(device_id)
        else:
            self.wrapped_newed = new CUDA(device_id, <unsigned> rng_seed)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __dealloc__(self):
        cdef CUDA *temp
        if self.wrapped_newed is not NULL:
            temp = <CUDA*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    @staticmethod
    def num_devices():
        return CUDA_num_devices()

    def dump_description(self):
        (<CUDA*> self.wrapped).dump_description()
        return

    def type(self):
        return (<CUDA*> self.wrapped).type()
