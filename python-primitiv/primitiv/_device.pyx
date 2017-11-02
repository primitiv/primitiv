from libc.stdint cimport uintptr_t

from weakref import WeakValueDictionary


# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_device_weak_dict = WeakValueDictionary()


cdef class _Device:

    @staticmethod
    def get_default():
        return _Device.get_wrapper(&CppDevice.get_default())

    @staticmethod
    def set_default(_Device dev):
        CppDevice.set_default(dev.wrapped[0])

    def dump_description(self):
        self.wrapped.dump_description()
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, _Device wrapper):
        if <uintptr_t> ptr in py_primitiv_device_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_device_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef _Device get_wrapper(CppDevice *ptr):
        # NOTE(vbkaisetsu):
        # _Device instances should be created and be registered before this
        # function is called.
        return py_primitiv_device_weak_dict[<uintptr_t> ptr]
