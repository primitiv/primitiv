from primitiv._tensor cimport CppTensor


cdef extern from "primitiv/initializer.h":
    cdef cppclass CppInitializer "primitiv::Initializer":
        CppInitializer() except +
        void apply(CppTensor &x) except +


cdef class Initializer:
    cdef CppInitializer *wrapped
    cdef CppInitializer *wrapped_newed


cdef inline Initializer wrapInitializer(CppInitializer *wrapped) except +:
    cdef Initializer initializer = Initializer.__new__(Initializer)
    initializer.wrapped = wrapped
    return initializer
