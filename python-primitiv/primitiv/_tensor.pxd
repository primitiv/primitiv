from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape


cdef extern from "primitiv/tensor.h" namespace "primitiv" nogil:
    cdef cppclass CppTensor "primitiv::Tensor":
        CppTensor(CppTensor &&src) except +
        CppTensor() except +
        bool valid() except +
        const CppShape &shape() except +
        CppDevice &device()
        const void *data() except +
        float to_float() except +
        vector[float] to_vector() except +
        void reset(const float k) except +
        # void reset_by_array(const float *values) except +
        void reset_by_vector(const vector[float] &values) except +
        CppTensor reshape(const CppShape &new_shape) except +
        CppTensor flatten() except +


cdef extern from "tensor_op.h":
    CppTensor &tensor_inplace_multiply_const(CppTensor &tensor, float k) except +
    CppTensor &tensor_inplace_add(CppTensor &tensor, const CppTensor &x) except +
    CppTensor &tensor_inplace_subtract(CppTensor &tensor, const CppTensor &x) except +


cdef class _Tensor:
    cdef CppTensor wrapped


cdef inline _Tensor wrapTensor(CppTensor wrapped) except +:
    cdef _Tensor tensor = _Tensor.__new__(_Tensor)
    tensor.wrapped = wrapped
    return tensor
