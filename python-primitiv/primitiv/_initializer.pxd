from primitiv._tensor cimport CppTensor


cdef extern from "primitiv/initializer.h" namespace "primitiv":
    cdef cppclass CppInitializer "primitiv::Initializer":
        CppInitializer() except +
        void apply(CppTensor &x) except +


cdef class _Initializer:
    cdef CppInitializer *wrapped
    cdef CppInitializer *wrapped_newed


cdef inline _Initializer wrapInitializer(CppInitializer *wrapped) except +:
    cdef _Initializer initializer = _Initializer.__new__(_Initializer)
    initializer.wrapped = wrapped
    return initializer
