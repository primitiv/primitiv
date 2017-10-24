from primitiv._initializer cimport CppInitializer, _Initializer


cdef extern from "primitiv/initializer_impl.h":
    cdef cppclass CppConstant "primitiv::initializers::Constant" (CppInitializer):
        CppConstant(float k)

    cdef cppclass CppUniform "primitiv::initializers::Uniform" (CppInitializer):
        CppUniform(float lower, float upper)

    cdef cppclass CppNormal "primitiv::initializers::Normal" (CppInitializer):
        CppNormal(float mean, float sd)

    cdef cppclass CppIdentity "primitiv::initializers::Identity" (CppInitializer):
        CppIdentity()

    cdef cppclass CppXavierUniform "primitiv::initializers::XavierUniform" (CppInitializer):
        CppXavierUniform(float scale)

    cdef cppclass CppXavierNormal "primitiv::initializers::XavierNormal" (CppInitializer):
        CppXavierNormal(float scale)


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
