from libcpp.vector cimport vector

from primitiv._shape cimport normShape
from primitiv._tensor cimport wrapTensor, _Tensor
from primitiv._device cimport get_default as Device_get_default
from primitiv._device cimport set_default as Device_set_default


cdef class _Device:

    @staticmethod
    def get_default():
        if py_primitiv_Device_default is None:
            raise RuntimeError("Default device is null.")
        return py_primitiv_Device_default

    @staticmethod
    def set_default(dev):
        global py_primitiv_Device_default
        Device_set_default((<_Device> dev).wrapped[0])
        py_primitiv_Device_default = dev

    def dump_description(self):
        self.wrapped.dump_description()
        return
