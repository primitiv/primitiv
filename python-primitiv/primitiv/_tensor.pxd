from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape


cdef extern from "primitiv/tensor.h" nogil:
    cdef cppclass CppTensor "primitiv::Tensor":
        CppTensor(CppTensor &&src) except +
        CppTensor() except +
        bool valid() except +
        const CppShape &shape() except +
        CppDevice &device()
        const void *data() except +
        float to_float() except +
        vector[float] to_vector() except +
        vector[unsigned] argmax(unsigned dim) except +
        vector[unsigned] argmin(unsigned dim) except +
        void reset(const float k) except +
        # void reset_by_array(const float *values) except +
        void reset_by_vector(const vector[float] &values) except +
        CppTensor reshape(const CppShape &new_shape) except +
        CppTensor flatten() except +


cdef extern from "tensor_op.h" namespace "python_primitiv_tensor":
    cdef CppTensor op_tensor_pos(const CppTensor &x) except +
    cdef CppTensor op_tensor_neg(const CppTensor &x) except +
    cdef CppTensor op_tensor_add(const CppTensor &x, float k) except +
    cdef CppTensor op_tensor_add(float k, const CppTensor &x) except +
    cdef CppTensor op_tensor_add(const CppTensor &a, const CppTensor &b) except +
    cdef CppTensor op_tensor_sub(const CppTensor &x, float k) except +
    cdef CppTensor op_tensor_sub(float k, const CppTensor &x) except +
    cdef CppTensor op_tensor_sub(const CppTensor &a, const CppTensor &b) except +
    cdef CppTensor op_tensor_mul(const CppTensor &x, float k) except +
    cdef CppTensor op_tensor_mul(float k, const CppTensor &x) except +
    cdef CppTensor op_tensor_mul(const CppTensor &a, const CppTensor &b) except +
    cdef CppTensor op_tensor_div(const CppTensor &x, float k) except +
    cdef CppTensor op_tensor_div(float k, const CppTensor &x) except +
    cdef CppTensor op_tensor_div(const CppTensor &a, const CppTensor &b) except +
    cdef void op_tensor_imul(CppTensor &tensor, float k) except +
    cdef void op_tensor_iadd(CppTensor &tensor, const CppTensor &x) except +
    cdef void op_tensor_isub(CppTensor &tensor, const CppTensor &x) except +


cdef class Tensor:
    cdef CppTensor *wrapped
    cdef bool del_required
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppTensor *ptr, Tensor wrapper)
    @staticmethod
    cdef Tensor get_wrapper(CppTensor *ptr)
    @staticmethod
    cdef Tensor get_wrapper_with_new(CppTensor *ptr)
