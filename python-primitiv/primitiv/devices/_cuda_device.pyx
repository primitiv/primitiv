from primitiv._device cimport _Device
from primitiv.devices._cuda_device cimport num_devices as CppCUDA_num_devices
from weakref import WeakValueDictionary
from libc.stdint cimport uintptr_t
from primitiv._device cimport py_primitiv_device_weak_dict


cdef class _CUDA(_Device):

    def __init__(self, unsigned device_id, rng_seed = None):
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if rng_seed == None:
            self.wrapped_newed = new CppCUDA(device_id)
        else:
            self.wrapped_newed = new CppCUDA(device_id, <unsigned> rng_seed)
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

        global py_primitiv_device_weak_dict
        if py_primitiv_device_weak_dict is None:
            py_primitiv_device_weak_dict = WeakValueDictionary()
        py_primitiv_device_weak_dict[<uintptr_t> self.wrapped_newed] = self

    def __dealloc__(self):
        cdef CppCUDA *temp
        if self.wrapped_newed is not NULL:
            temp = <CppCUDA*> self.wrapped_newed
            del temp
            self.wrapped_newed = NULL

    @staticmethod
    def num_devices():
        return CppCUDA_num_devices()
