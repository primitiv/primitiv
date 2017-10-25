from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uintptr_t

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


cdef class _Tensor:
    cdef CppTensor *wrapped
    cdef bool del_required
    cdef object __weakref__


# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_tensor_weak_dict

cdef inline _Tensor wrapTensor(CppTensor *wrapped) except +:
    cdef _Tensor tensor
    global py_primitiv_tensor_weak_dict
    if py_primitiv_tensor_weak_dict is None:
        from weakref import WeakValueDictionary
        py_primitiv_tensor_weak_dict = WeakValueDictionary()
    ret = py_primitiv_tensor_weak_dict.get(<uintptr_t> wrapped)
    if ret:
        return ret
    tensor = _Tensor.__new__(_Tensor)
    tensor.wrapped = wrapped
    tensor.del_required = False
    py_primitiv_tensor_weak_dict[<uintptr_t> wrapped] = tensor
    return tensor


cdef inline _Tensor wrapTensorWithNew(CppTensor *wrapped) except +:
    global py_primitiv_tensor_weak_dict
    cdef _Tensor tensor = _Tensor.__new__(_Tensor)
    tensor.wrapped = wrapped
    if py_primitiv_tensor_weak_dict is None:
        from weakref import WeakValueDictionary
        py_primitiv_tensor_weak_dict = WeakValueDictionary()
    tensor.del_required = True
    py_primitiv_tensor_weak_dict[<uintptr_t> tensor.wrapped] = tensor
    return tensor
