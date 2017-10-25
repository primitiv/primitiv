from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t

from primitiv._shape cimport normShape
from primitiv._tensor cimport _Tensor

from weakref import WeakValueDictionary

# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_device_weak_dict = WeakValueDictionary()


cdef class _Device:

    @staticmethod
    def get_default():
        return _Device.get_wrapper(&CppDevice_get_default())

    @staticmethod
    def set_default(_Device dev):
        CppDevice_set_default(dev.wrapped[0])

    def dump_description(self):
        self.wrapped.dump_description()
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, _Device wrapper):
        global py_primitiv_device_weak_dict
        py_primitiv_device_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef _Device get_wrapper(CppDevice *ptr):
        global py_primitiv_device_weak_dict
        # _Device instances should be created and be registered before this
        # function is called.
        return py_primitiv_device_weak_dict[<uintptr_t> ptr]
