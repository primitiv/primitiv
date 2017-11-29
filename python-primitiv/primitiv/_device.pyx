from libc.stdint cimport uintptr_t

from weakref import WeakValueDictionary


# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_device_weak_dict = WeakValueDictionary()


cdef class Device:
    """Interface of the Tensor provider.

    """

    @staticmethod
    def get_default():
        """Retrieves the current default device.

        :return: Reference of the current default device.
        :rtype: primitiv.Device
        :raises RuntimeError: if the default device is null.

        """
        return Device.get_wrapper(&CppDevice.get_default())

    @staticmethod
    def set_default(Device dev):
        """Specifies a new default device.

        :param dev: Reference of the new default device.
        :type dev: primitiv.Device

        """
        CppDevice.set_default(dev.wrapped[0])

    def dump_description(self):
        """Prints device description to stderr.

        """
        self.wrapped.dump_description()
        return

    def __copy__(self):
        raise NotImplementedError(type(self).__name__ + " does not support `__copy__` for now.")

    def __deepcopy__(self, memo):
        raise NotImplementedError(type(self).__name__ + " does not support `__deepcopy__` for now.")

    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, Device wrapper):
        if <uintptr_t> ptr in py_primitiv_device_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_device_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef Device get_wrapper(CppDevice *ptr):
        # NOTE(vbkaisetsu):
        # Device instances should be created and be registered before this
        # function is called.
        return py_primitiv_device_weak_dict[<uintptr_t> ptr]
