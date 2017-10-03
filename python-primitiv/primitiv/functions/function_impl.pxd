from libcpp.vector cimport vector
from libcpp.string cimport string

from primitiv.tensor cimport Tensor, _Tensor
from primitiv.shape cimport Shape, _Shape
from primitiv.device cimport Device
from primitiv.function cimport Function, _Function
from primitiv.parameter cimport Parameter


cdef extern from "primitiv/function_impl.h" namespace "primitiv::functions":
    cdef cppclass Input(Function):
        Input(const Shape &shape, const vector[float] &data, Device &device);

    cdef cppclass ParameterInput(Function):
        ParameterInput(Parameter &param)

    cdef cppclass Copy(Function):
        Copy(Device &device)

    cdef cppclass Constant(Function):
        Constant(const Shape &shape, float k, Device &device)

    cdef cppclass IdentityMatrix(Function):
        IdentityMatrix(unsigned size, Device &device)

    cdef cppclass RandomBernoulli(Function):
        RandomBernoulli(const Shape &shape, float p, Device &device)

    cdef cppclass RandomUniform(Function):
        RandomUniform(const Shape &shape, float lower, float upper, Device &device)

    cdef cppclass RandomNormal(Function):
        RandomNormal(const Shape &shape, float mean, float sd, Device &device)

    cdef cppclass RandomLogNormal(Function):
        RandomLogNormal(const Shape &shape, float mean, float sd, Device &device)

    cdef cppclass Pick(Function):
        Pick(const vector[unsigned] &ids, unsigned dim)

    cdef cppclass Slice(Function):
        Slice(unsigned dim, unsigned lower, unsigned upper)

    cdef cppclass Concat(Function):
        Concat(unsigned dim)

    cdef cppclass Reshape(Function):
        Reshape(const Shape &shape)

    cdef cppclass Sum(Function):
        Sum(unsigned dim)

    cdef cppclass LogSumExp(Function):
        LogSumExp(unsigned dim)

    cdef cppclass Broadcast(Function):
        Broadcast(unsigned dim, unsigned size)

    cdef cppclass SoftmaxCrossEntropy(Function):
        SoftmaxCrossEntropy(unsigned dim)

    cdef cppclass SparseSoftmaxCrossEntropy(Function):
        SparseSoftmaxCrossEntropy(const vector[unsigned] ids, unsigned dim)


cdef class _Input(_Function):
    pass

cdef class _ParameterInput(_Function):
    pass

cdef class _Copy(_Function):
    pass

cdef class _Constant(_Function):
    pass

cdef class _IdentityMatrix(_Function):
    pass

cdef class _RandomBernoulli(_Function):
    pass

cdef class _RandomUniform(_Function):
    pass

cdef class _RandomNormal(_Function):
    pass

cdef class _RandomLogNormal(_Function):
    pass

cdef class _Pick(_Function):
    pass

cdef class _Slice(_Function):
    pass

cdef class _Concat(_Function):
    pass

cdef class _Reshape(_Function):
    pass

cdef class _Sum(_Function):
    pass

cdef class _LogSumExp(_Function):
    pass

cdef class _Broadcast(_Function):
    pass

cdef class _SoftmaxCrossEntropy(_Function):
    pass

cdef class _SparseSoftmaxCrossEntropy(_Function):
    pass
