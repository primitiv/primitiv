from primitiv._tensor cimport Tensor


cdef extern from "primitiv/initializer.h" namespace "primitiv":
    cdef cppclass Initializer:
        Initializer() except +
        void apply(Tensor &x) except +


cdef class _Initializer:
    cdef Initializer *wrapped
    cdef Initializer *wrapped_newed


cdef inline _Initializer wrapInitializer(Initializer *wrapped) except +:
    cdef _Initializer initializer = _Initializer.__new__(_Initializer)
    initializer.wrapped = wrapped
    return initializer
