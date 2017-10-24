from primitiv._device cimport _Device
from weakref import WeakValueDictionary
from libc.stdint cimport uintptr_t
from primitiv._device cimport py_primitiv_device_weak_dict


cdef class _Naive(_Device):

    def __init__(self, rng_seed = None):
        if self.wrapped is not NULL:
            raise MemoryError()
        if rng_seed is None:
            self.wrapped = new CppNaive()
        else:
            self.wrapped = new CppNaive(<unsigned> rng_seed)

        global py_primitiv_device_weak_dict
        if py_primitiv_device_weak_dict is None:
            py_primitiv_device_weak_dict = WeakValueDictionary()
        py_primitiv_device_weak_dict[<uintptr_t> self.wrapped] = self

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL
