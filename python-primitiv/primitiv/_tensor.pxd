from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport Device
from primitiv._shape cimport Shape


cdef extern from "primitiv/tensor.h" namespace "primitiv" nogil:
    cdef cppclass Tensor:
        Tensor(Tensor &&src) except +
        Tensor() except +
        bool valid() except +
        const Shape &shape() except +
        Device &device()
        const void *data() except +
        vector[float] to_vector() except +
        void reset(const float k) except +
        # void reset_by_array(const float *values) except +
        void reset_by_vector(const vector[float] &values) except +
        Tensor reshape(const Shape &new_shape) except +
        Tensor flatten() except +


cdef extern from "tensor_op.h":
    Tensor &tensor_inplace_multiply_const(Tensor &tensor, float k) except +
    Tensor &tensor_inplace_add(Tensor &tensor, const Tensor &x) except +
    Tensor &tensor_inplace_subtract(Tensor &tensor, const Tensor &x) except +


cdef class _Tensor:
    cdef Tensor wrapped


cdef inline _Tensor wrapTensor(Tensor wrapped) except +:
    cdef _Tensor tensor = _Tensor.__new__(_Tensor)
    tensor.wrapped = wrapped
    return tensor
