from libcpp.vector cimport vector

from primitiv.shape cimport Shape, _Shape, wrapShape
from primitiv.tensor cimport wrapTensor, _Tensor
from primitiv.device cimport wrapDevice, _Device
from primitiv.function cimport _Function
from primitiv.parameter cimport _Parameter

import numpy as np

from ..utils cimport ndarray_to_vector


cdef class _Input(_Function):
    """
    Input(shape, data, device):
     or
    Input(data, device):
    """
    def __cinit__(self, *args):
        if len(args) == 3:
            self.wrapped = new Input((<_Shape> args[0]).wrapped, args[1], (<_Device> args[2]).wrapped[0])
            if self.wrapped is NULL:
                raise MemoryError()
        elif len(args) == 2:
            shape = _Shape(args[0].shape)
            self.wrapped = new Input(shape.wrapped, ndarray_to_vector(args[0]), (<_Device> args[1]).wrapped[0])
            if self.wrapped is NULL:
                raise MemoryError()
        else:
            raise TypeError("Input() takes two or three arguments (%d given)" % len(args))

    def __dealloc__(self):
        cdef Input *temp
        if self.wrapped is not NULL:
            temp = <Input*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<Input*> self.wrapped).get_device())

    def name(self):
        return (<Input*> self.wrapped).name().decode("utf-8")


cdef class _ParameterInput(_Function):
    def __cinit__(self, _Parameter param):
        self.wrapped = new ParameterInput(param.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef ParameterInput *temp
        if self.wrapped is not NULL:
            temp = <ParameterInput*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<ParameterInput*> self.wrapped).get_device())

    def get_inner_value(self):
        return wrapTensor((<ParameterInput*> self.wrapped).get_inner_value()[0])

    def name(self):
        return (<ParameterInput*> self.wrapped).name().decode("utf-8")


cdef class _Copy(_Function):
    def __init__(self, _Device device):
        self.wrapped = new Copy(device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Copy *temp
        if self.wrapped is not NULL:
            temp = <Copy*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<Copy*> self.wrapped).get_device())

    def name(self):
        return (<Copy*> self.wrapped).name().decode("utf-8")


cdef class _Constant(_Function):
    def __cinit__(self, _Shape shape, float k, _Device device):
        self.wrapped = new Constant(shape.wrapped, k, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Constant *temp
        if self.wrapped is not NULL:
            temp = <Constant*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<Constant*> self.wrapped).get_device())

    def name(self):
        return (<Constant*> self.wrapped).name().decode("utf-8")


cdef class _IdentityMatrix(_Function):
    def __cinit__(self, unsigned size, _Device device):
        self.wrapped = new IdentityMatrix(size, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef IdentityMatrix *temp
        if self.wrapped is not NULL:
            temp = <IdentityMatrix*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<IdentityMatrix*> self.wrapped).get_device())

    def name(self):
        return (<IdentityMatrix*> self.wrapped).name().decode("utf-8")


cdef class _RandomBernoulli(_Function):
    def __cinit__(self, _Shape shape, float p, _Device device):
        self.wrapped = new RandomBernoulli(shape.wrapped, p, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef RandomBernoulli *temp
        if self.wrapped is not NULL:
            temp = <RandomBernoulli*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<RandomBernoulli*> self.wrapped).get_device())

    def name(self):
        return (<RandomBernoulli*> self.wrapped).name().decode("utf-8")


cdef class _RandomUniform(_Function):
    def __cinit__(self, _Shape shape, float lower, float upper, _Device device):
        self.wrapped = new RandomUniform(shape.wrapped, lower, upper, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef RandomUniform *temp
        if self.wrapped is not NULL:
            temp = <RandomUniform*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<RandomUniform*> self.wrapped).get_device())

    def name(self):
        return (<RandomUniform*> self.wrapped).name().decode("utf-8")


cdef class _RandomNormal(_Function):
    def __cinit__(self, _Shape shape, float mean, float sd, _Device device):
        self.wrapped = new RandomNormal(shape.wrapped, mean, sd, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef RandomNormal *temp
        if self.wrapped is not NULL:
            temp = <RandomNormal*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<RandomNormal*> self.wrapped).get_device())

    def name(self):
        return (<RandomNormal*> self.wrapped).name().decode("utf-8")


cdef class _RandomLogNormal(_Function):
    def __cinit__(self, _Shape shape, float mean, float sd, _Device device):
        self.wrapped = new RandomLogNormal(shape.wrapped, mean, sd, device.wrapped[0])
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef RandomLogNormal *temp
        if self.wrapped is not NULL:
            temp = <RandomLogNormal*> self.wrapped
            del temp
            self.wrapped = NULL

    def get_device(self):
        return wrapDevice((<RandomLogNormal*> self.wrapped).get_device())

    def name(self):
        return (<RandomLogNormal*> self.wrapped).name().decode("utf-8")


cdef class _Pick(_Function):
    def __cinit__(self, vector[unsigned] ids, unsigned dim):
        self.wrapped = new Pick(ids, dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Pick *temp
        if self.wrapped is not NULL:
            temp = <Pick*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Pick*> self.wrapped).name().decode("utf-8")


cdef class _Slice(_Function):
    def __cinit__(self, unsigned dim, unsigned lower, unsigned upper):
        self.wrapped = new Slice(dim, lower, upper)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Slice *temp
        if self.wrapped is not NULL:
            temp = <Slice*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Slice*> self.wrapped).name().decode("utf-8")


cdef class _Concat(_Function):
    def __cinit__(self, unsigned dim):
        self.wrapped = new Concat(dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Concat *temp
        if self.wrapped is not NULL:
            temp = <Concat*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Concat*> self.wrapped).name().decode("utf-8")


cdef class _Reshape(_Function):
    def __cinit__(self, _Shape shape):
        self.wrapped = new Reshape(shape.wrapped)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Reshape *temp
        if self.wrapped is not NULL:
            temp = <Reshape*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Reshape*> self.wrapped).name().decode("utf-8")


cdef class _Sum(_Function):
    def __cinit__(self, unsigned dim):
        self.wrapped = new Sum(dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Sum *temp
        if self.wrapped is not NULL:
            temp = <Sum*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Sum*> self.wrapped).name().decode("utf-8")


cdef class _LogSumExp(_Function):
    def __cinit__(self, unsigned dim):
        self.wrapped = new LogSumExp(dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef LogSumExp *temp
        if self.wrapped is not NULL:
            temp = <LogSumExp*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<LogSumExp*> self.wrapped).name().decode("utf-8")


cdef class _Broadcast(_Function):
    def __cinit__(self, unsigned dim, unsigned size):
        self.wrapped = new Broadcast(dim, size)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef Broadcast *temp
        if self.wrapped is not NULL:
            temp = <Broadcast*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<Broadcast*> self.wrapped).name().decode("utf-8")


cdef class _SoftmaxCrossEntropy(_Function):
    def __cinit__(self, unsigned dim):
        self.wrapped = new SoftmaxCrossEntropy(dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef SoftmaxCrossEntropy *temp
        if self.wrapped is not NULL:
            temp = <SoftmaxCrossEntropy*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<SoftmaxCrossEntropy*> self.wrapped).name().decode("utf-8")


cdef class _SparseSoftmaxCrossEntropy(_Function):
    def __cinit__(self, vector[unsigned] ids, unsigned dim):
        self.wrapped = new SparseSoftmaxCrossEntropy(ids, dim)
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        cdef SparseSoftmaxCrossEntropy *temp
        if self.wrapped is not NULL:
            temp = <SparseSoftmaxCrossEntropy*> self.wrapped
            del temp
            self.wrapped = NULL

    def name(self):
        return (<SparseSoftmaxCrossEntropy*> self.wrapped).name().decode("utf-8")
