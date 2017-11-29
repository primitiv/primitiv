from primitiv._initializer cimport CppInitializer, Initializer


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


cdef class Constant(Initializer):
    pass

cdef class Uniform(Initializer):
    pass

cdef class Normal(Initializer):
    pass

cdef class Identity(Initializer):
    pass

cdef class XavierUniform(Initializer):
    pass

cdef class XavierNormal(Initializer):
    pass
