cdef extern from "primitiv/device.h":
    cdef cppclass CppDevice "primitiv::Device":
        @staticmethod
        CppDevice &get_default() except +
        @staticmethod
        void set_default(CppDevice &dev) except +
        void dump_description() except +


cdef class Device:
    cdef CppDevice *wrapped
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, Device wrapper)
    @staticmethod
    cdef Device get_wrapper(CppDevice *ptr)
