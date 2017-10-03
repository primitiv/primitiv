from libcpp.vector cimport vector
from libcpp.string cimport string

from primitiv.tensor cimport Tensor, _Tensor
from primitiv.shape cimport Shape, _Shape
from primitiv.device cimport Device

import cython.template


cdef extern from "primitiv/function.h" namespace "primitiv":
    cdef cppclass Function:
        Shape forward_shape(vector[const Shape *] &args) except +
        Device *get_device() except +
        const Tensor *get_inner_value() except +
        Tensor forward(const vector[const Tensor *] &args) except +
        void backward(
            const Tensor &cur_value,
            const Tensor &cur_grad,
            const vector[const Tensor *] &arg_values,
            const vector[Tensor *] &arg_grads) except +
        string name() except +


cdef class _Function:
    cdef Function *wrapped


cdef inline _Function wrapFunction(Function *wrapped) except +:
    cdef _Function function = _Function.__new__(_Function)
    function.wrapped = wrapped
    return function
