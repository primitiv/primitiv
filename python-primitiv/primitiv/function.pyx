from libcpp.vector cimport vector

from primitiv.shape cimport Shape, wrapShape, normShape
from primitiv.tensor cimport wrapTensor, _Tensor
from primitiv.device cimport wrapDevice


cdef class _Function:

    def forward_shape(self, args):
        cdef vector[const Shape *] vec
        for x in args:
            vec.push_back(&normShape(x).wrapped)
        return wrapShape(self.wrapped.forward_shape(vec))

    def get_device(self):
        return wrapDevice(self.wrapped.get_device())

    def get_inner_value(self):
        cdef const Tensor *tensor = self.wrapped.get_inner_value()
        if tensor == NULL:
            return None
        return wrapTensor(tensor[0])

    def forward(self, args):
        cdef vector[const Tensor *] vec
        cdef _Tensor x
        for x in args:
            vec.push_back(&x.wrapped)
        return wrapTensor(self.wrapped.forward(vec))

    def backward(self, _Tensor cur_value, _Tensor cur_grad, arg_values, arg_grads):
        cdef vector[const Tensor *] vec_arg_values
        cdef vector[Tensor *] vec_arg_grads
        cdef _Tensor x
        for x in arg_values:
            vec_arg_values.push_back(&x.wrapped)
        for x in arg_grads:
            vec_arg_grads.push_back(&x.wrapped)
        self.wrapped.backward(cur_value.wrapped, cur_grad.wrapped, vec_arg_values, vec_arg_grads)
        return

    def name(self):
        return self.wrapped.name().decode("utf-8")
