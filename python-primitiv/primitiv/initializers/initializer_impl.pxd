from primitiv.initializer cimport Initializer, _Initializer


cdef extern from "primitiv/initializer_impl.h" namespace "primitiv::initializers":
    cdef cppclass Constant(Initializer):
        Constant(float k)

    cdef cppclass Uniform(Initializer):
        Uniform(float lower, float upper)

    cdef cppclass Normal(Initializer):
        Normal(float mean, float sd)

    cdef cppclass Identity(Initializer):
        Identity()

    cdef cppclass XavierUniform(Initializer):
        XavierUniform(float scale)

    cdef cppclass XavierNormal(Initializer):
        XavierNormal(float scale)


cdef class _Constant(_Initializer):
    pass

cdef class _Uniform(_Initializer):
    pass

cdef class _Normal(_Initializer):
    pass

cdef class _Identity(_Initializer):
    pass

cdef class _XavierUniform(_Initializer):
    pass

cdef class _XavierNormal(_Initializer):
    pass
