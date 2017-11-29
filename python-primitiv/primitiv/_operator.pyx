from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport Device
from primitiv._shape cimport Shape, normShape
from primitiv._tensor cimport Tensor, CppTensor
from primitiv._graph cimport Graph, wrapNode, CppNode, Node
from primitiv._parameter cimport Parameter

from utils cimport ndarrays_to_vector

cimport numpy as np
import numpy as np


class operators:

    @staticmethod
    def raw_input(shape, vector[float] data, Device device = None, Graph g = None):
        if device is None:
            device = Device.get_default()
        if g is not None:
            return wrapNode(Node_input_vector(normShape(shape).wrapped, data, device.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_input_vector(normShape(shape).wrapped, data, device.wrapped[0]))

    # NOTE(vbkaisetsu)
    # This function takes an np.ndarray or a list of np.ndarray
    # instead of a vector.
    @staticmethod
    def input(data, Device device = None, Graph g = None):
        # NOTE(vbkaisetsu, odashi):
        # In this function, we don't check whether each ndarray is empty
        # (i.e., it doesn't have any elements) or not.
        # When the ndarray contains no element, its shape becomes (0,),
        # and primitiv.Shape will reject the shape and raises an exception.
        # In addition, we also don't check whether each ndarray has the same shape or not.
        # This condition will be checked in ndarrays_to_vector().
        if isinstance(data, np.ndarray):
            data = [data]
        if isinstance(data, list):
            if len(data) == 0:
                raise TypeError("`data` contains no item.")
            if not isinstance(data[0], np.ndarray):
                raise TypeError("`data` contains other objects than numpy.ndarray.")
            shape = Shape(data[0].shape, len(data))
        else:
            raise TypeError("`data` has incorrect type.")
        return operators.raw_input(shape, ndarrays_to_vector(data), device, g)


    @staticmethod
    def parameter(Parameter param, Graph g = None):
        if g is not None:
            return wrapNode(Node_parameter(param.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_parameter(param.wrapped[0]))

    @staticmethod
    def copy(Node x, Device device = None):
        if device is None:
            device = Device.get_default()
        return wrapNode(op_copy(x.wrapped, device.wrapped[0]))

    @staticmethod
    def pick(Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(op_pick(x.wrapped, ids, dim))

    @staticmethod
    def slice(Node x, unsigned dim, unsigned lower, unsigned upper):
        return wrapNode(op_slice(x.wrapped, dim, lower, upper))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppNode] vec
        cdef Node x
        for x in xs:
            vec.push_back(x.wrapped)
        return wrapNode(op_concat(vec, dim))

    @staticmethod
    def reshape(Node x, Shape new_shape):
        return wrapNode(op_reshape(x.wrapped, new_shape.wrapped))

    @staticmethod
    def flatten(Node x):
        return wrapNode(op_flatten(x.wrapped))

    @staticmethod
    def transpose(Node x):
        return wrapNode(op_transpose(x.wrapped))

    @staticmethod
    def matmul(Node a, Node b):
        return wrapNode(op_matmul(a.wrapped, b.wrapped))

    @staticmethod
    def sqrt(Node x):
        return wrapNode(op_sqrt(x.wrapped))

    @staticmethod
    def exp(Node x):
        return wrapNode(op_exp(x.wrapped))

    @staticmethod
    def log(Node x):
        return wrapNode(op_log(x.wrapped))

    @staticmethod
    def pow(x, k):
        if isinstance(x, Node) and isinstance(k, int) and -0x80000000 <= k <= 0x7fffffff:
            return wrapNode(op_ipow((<Node> x).wrapped, <int> k))
        elif isinstance(x, Node) and isinstance(k, (int, float)):
            return wrapNode(op_pow((<Node> x).wrapped, <float> k))
        elif isinstance(x, (int, float)) and isinstance(k, Node):
            return wrapNode(op_pow(<float> x, (<Node> k).wrapped))
        elif isinstance(x, Node) and isinstance(k, Node):
            return wrapNode(op_pow((<Node> x).wrapped, (<Node> k).wrapped))
        else:
            raise TypeError("`x` or `k` has incorrect type.")

    @staticmethod
    def tanh(Node x):
        return wrapNode(op_tanh(x.wrapped))

    @staticmethod
    def sigmoid(Node x):
        return wrapNode(op_sigmoid(x.wrapped))

    @staticmethod
    def softplus(Node x):
        return wrapNode(op_softplus(x.wrapped))

    @staticmethod
    def sin(Node x):
        return wrapNode(op_sin(x.wrapped))

    @staticmethod
    def cos(Node x):
        return wrapNode(op_cos(x.wrapped))

    @staticmethod
    def tan(Node x):
        return wrapNode(op_tan(x.wrapped))

    @staticmethod
    def relu(Node x):
        return wrapNode(op_relu(x.wrapped))

    @staticmethod
    def lrelu(Node x):
        return wrapNode(op_lrelu(x.wrapped))

    @staticmethod
    def prelu(Node x, float a):
        return wrapNode(op_prelu(x.wrapped, a))

    @staticmethod
    def elu(Node x, float a):
        return wrapNode(op_elu(x.wrapped, a))

    @staticmethod
    def selu(Node x, float a, float s):
        return wrapNode(op_selu(x.wrapped, a, s))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppNode] xs
        cdef Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(Node_sum_container(xs))
        else:
            return wrapNode(Node_sum((<Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppNode] xs
        cdef Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(Node_mean_container(xs))
        else:
            return wrapNode(Node_mean((<Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def broadcast(Node x, unsigned dim, unsigned size):
        return wrapNode(op_broadcast(x.wrapped, dim, size))

    @staticmethod
    def logsumexp(Node x, unsigned dim):
        return wrapNode(op_logsumexp(x.wrapped, dim))

    @staticmethod
    def log_softmax(Node x, unsigned dim):
        return wrapNode(op_log_softmax(x.wrapped, dim))

    @staticmethod
    def softmax(Node x, unsigned dim):
        return wrapNode(op_softmax(x.wrapped, dim))

    @staticmethod
    def softmax_cross_entropy(Node x, t, unsigned dim):
        if isinstance(t, Node):
            return wrapNode(op_softmax_cross_entropy(x.wrapped, (<Node> t).wrapped, dim))
        elif isinstance(t, list):
            return wrapNode(op_softmax_cross_entropy(x.wrapped, <vector[unsigned]> t, dim))
        else:
            raise TypeError("`t` has incorrect type.")

    @staticmethod
    def constant(shape, float k, Device device = None, Graph g = None):
        if device is None:
            device = Device.get_default()
        if g is None:
            return wrapNode(op_constant[CppNode](normShape(shape).wrapped, k, device.wrapped[0]))
        else:
            return wrapNode(op_constant(normShape(shape).wrapped, k, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def zeros(shape, Device device = None, Graph g = None):
        if device is None:
            device = Device.get_default()
        if g is None:
            return wrapNode(op_zeros[CppNode](normShape(shape).wrapped, device.wrapped[0]))
        else:
            return wrapNode(op_zeros(normShape(shape).wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def ones(shape, Device device = None, Graph g = None):
        if device is None:
            device = Device.get_default()
        if g is None:
            return wrapNode(op_ones[CppNode](normShape(shape).wrapped, device.wrapped[0]))
        else:
            return wrapNode(op_ones(normShape(shape).wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def identity(unsigned size, Device device = None, Graph g = None):
        if device is None:
            device = Device.get_default()
        if g is None:
            return wrapNode(op_identity[CppNode](size, device.wrapped[0]))
        else:
            return wrapNode(op_identity(size, device.wrapped[0], g.wrapped[0]))

    class batch:
        @staticmethod
        def sum(Node x):
            return wrapNode(op_batch_sum[CppNode](x.wrapped))

        @staticmethod
        def mean(Node x):
            return wrapNode(op_batch_mean[CppNode](x.wrapped))

        @staticmethod
        def normalize(Node x):
            return wrapNode(op_batch_normalize[CppNode](x.wrapped))

    class random:
        @staticmethod
        def bernoulli(shape, float p, Device device = None, Graph g = None):
            if device is None:
                device = Device.get_default()
            if g is None:
                return wrapNode(op_random_bernoulli[CppNode](normShape(shape).wrapped, p, device.wrapped[0]))
            else:
                return wrapNode(op_random_bernoulli(normShape(shape).wrapped, p, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def uniform(shape, float lower, float upper, Device device = None, Graph g = None):
            if device is None:
                device = Device.get_default()
            if g is None:
                return wrapNode(op_random_uniform[CppNode](normShape(shape).wrapped, lower, upper, device.wrapped[0]))
            else:
                return wrapNode(op_random_uniform(normShape(shape).wrapped, lower, upper, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def normal(shape, float mean, float sd, Device device = None, Graph g = None):
            if device is None:
                device = Device.get_default()
            if g is None:
                return wrapNode(op_random_normal[CppNode](normShape(shape).wrapped, mean, sd, device.wrapped[0]))
            else:
                return wrapNode(op_random_normal(normShape(shape).wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def log_normal(shape, float mean, float sd, Device device = None, Graph g = None):
            if device is None:
                device = Device.get_default()
            if g is None:
                return wrapNode(op_random_log_normal[CppNode](normShape(shape).wrapped, mean, sd, device.wrapped[0]))
            else:
                return wrapNode(op_random_log_normal(normShape(shape).wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def gumbel(shape, float mu, float beta, Device device = None, Graph g = None):
            if device is None:
                device = Device.get_default()
            if g is None:
                return wrapNode(op_random_gumbel[CppNode](normShape(shape).wrapped, mu, beta, device.wrapped[0]))
            else:
                return wrapNode(op_random_gumbel(normShape(shape).wrapped, mu, beta, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def dropout(Node x, float rate, bool enabled):
        return wrapNode(op_dropout(x.wrapped, rate, enabled))


class tensor_operators:

    @staticmethod
    def raw_input(shape, vector[float] data, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(Tensor_input_vector(normShape(shape).wrapped, data, device.wrapped[0])))

    # NOTE(vbkaisetsu)
    # This function takes an np.ndarray or a list of np.ndarray
    # instead of a vector.
    @staticmethod
    def input(data, Device device = None):
        # NOTE(vbkaisetsu, odashi):
        # In this function, we don't check whether each ndarray is empty
        # (i.e., it doesn't have any elements) or not.
        # When the ndarray contains no element, its shape becomes (0,),
        # and primitiv.Shape will reject the shape and raises an exception.
        # In addition, we also don't check whether each ndarray has the same shape or not.
        # This condition will be checked in ndarrays_to_vector().
        if isinstance(data, np.ndarray):
            data = [data]
        if isinstance(data, list):
            if len(data) == 0:
                raise TypeError("`data` contains no item.")
            if not isinstance(data[0], np.ndarray):
                raise TypeError("`data` contains other objects than numpy.ndarray.")
            shape = Shape(data[0].shape, len(data))
        else:
            raise TypeError("`data` has incorrect type.")
        return tensor_operators.raw_input(shape, ndarrays_to_vector(data), device)

    @staticmethod
    def parameter(Parameter param):
        return Tensor.get_wrapper_with_new(new CppTensor(Tensor_parameter(param.wrapped[0])))

    @staticmethod
    def copy(Tensor x, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(op_copy(x.wrapped[0], device.wrapped[0])))

    @staticmethod
    def pick(Tensor x, vector[unsigned] ids, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(op_pick(x.wrapped[0], ids, dim)))

    @staticmethod
    def slice(Tensor x, unsigned dim, unsigned lower, unsigned upper):
        return Tensor.get_wrapper_with_new(new CppTensor(op_slice(x.wrapped[0], dim, lower, upper)))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppTensor] vec
        cdef Tensor x
        for x in xs:
            vec.push_back(x.wrapped[0])
        return Tensor.get_wrapper_with_new(new CppTensor(op_concat(vec, dim)))

    @staticmethod
    def reshape(Tensor x, Shape new_shape):
        return Tensor.get_wrapper_with_new(new CppTensor(op_reshape(x.wrapped[0], new_shape.wrapped)))

    @staticmethod
    def flatten(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_flatten(x.wrapped[0])))

    @staticmethod
    def transpose(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_transpose(x.wrapped[0])))

    @staticmethod
    def matmul(Tensor a, Tensor b):
        return Tensor.get_wrapper_with_new(new CppTensor(op_matmul(a.wrapped[0], b.wrapped[0])))

    @staticmethod
    def sqrt(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_sqrt(x.wrapped[0])))

    @staticmethod
    def exp(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_exp(x.wrapped[0])))

    @staticmethod
    def log(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_log(x.wrapped[0])))

    @staticmethod
    def pow(x, k):
        if isinstance(x, Tensor) and isinstance(k, int) and -0x80000000 <= k <= 0x7fffffff:
            return Tensor.get_wrapper_with_new(new CppTensor(op_ipow((<Tensor> x).wrapped[0], <int> k)))
        elif isinstance(x, Tensor) and isinstance(k, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow((<Tensor> x).wrapped[0], <float> k)))
        elif isinstance(x, (int, float)) and isinstance(k, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow(<float> x, (<Tensor> k).wrapped[0])))
        elif isinstance(x, Tensor) and isinstance(k, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_pow((<Tensor> x).wrapped[0], (<Tensor> k).wrapped[0])))
        else:
            raise TypeError("`x` or `k` has incorrect type.")

    @staticmethod
    def tanh(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_tanh(x.wrapped[0])))

    @staticmethod
    def sigmoid(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_sigmoid(x.wrapped[0])))

    @staticmethod
    def softplus(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_softplus(x.wrapped[0])))

    @staticmethod
    def sin(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_sin(x.wrapped[0])))

    @staticmethod
    def cos(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_cos(x.wrapped[0])))

    @staticmethod
    def tan(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_tan(x.wrapped[0])))

    @staticmethod
    def relu(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_relu(x.wrapped[0])))

    @staticmethod
    def lrelu(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(op_lrelu(x.wrapped[0])))

    @staticmethod
    def prelu(Tensor x, float a):
        return Tensor.get_wrapper_with_new(new CppTensor(op_prelu(x.wrapped[0], a)))

    @staticmethod
    def elu(Tensor x, float a):
        return Tensor.get_wrapper_with_new(new CppTensor(op_elu(x.wrapped[0], a)))

    @staticmethod
    def selu(Tensor x, float a, float s):
        return Tensor.get_wrapper_with_new(new CppTensor(op_selu(x.wrapped[0], a, s)))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppTensor] xs
        cdef Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return Tensor.get_wrapper_with_new(new CppTensor(Tensor_sum_container(xs)))
        else:
            return Tensor.get_wrapper_with_new(new CppTensor(Tensor_sum((<Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppTensor] xs
        cdef Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return Tensor.get_wrapper_with_new(new CppTensor(Tensor_mean_container(xs)))
        else:
            return Tensor.get_wrapper_with_new(new CppTensor(Tensor_mean((<Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def broadcast(Tensor x, unsigned dim, unsigned size):
        return Tensor.get_wrapper_with_new(new CppTensor(op_broadcast(x.wrapped[0], dim, size)))

    @staticmethod
    def logsumexp(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(op_logsumexp(x.wrapped[0], dim)))

    @staticmethod
    def log_softmax(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(op_log_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(op_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax_cross_entropy(Tensor x, t, unsigned dim):
        if isinstance(t, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(op_softmax_cross_entropy(x.wrapped[0], (<Tensor> t).wrapped[0], dim)))
        elif isinstance(t, list):
            return Tensor.get_wrapper_with_new(new CppTensor(op_softmax_cross_entropy(x.wrapped[0], <vector[unsigned]> t, dim)))
        else:
            raise TypeError("`t` has incorrect type.")

    @staticmethod
    def constant(shape, float k, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(op_constant[CppTensor](normShape(shape).wrapped, k, device.wrapped[0])))

    @staticmethod
    def zeros(shape, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(op_zeros[CppTensor](normShape(shape).wrapped, device.wrapped[0])))

    @staticmethod
    def ones(shape, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(op_ones[CppTensor](normShape(shape).wrapped, device.wrapped[0])))

    @staticmethod
    def identity(unsigned size, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(op_identity[CppTensor](size, device.wrapped[0])))

    class batch:
        @staticmethod
        def sum(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(op_batch_sum[CppTensor](x.wrapped[0])))

        @staticmethod
        def mean(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(op_batch_mean[CppTensor](x.wrapped[0])))

        @staticmethod
        def normalize(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(op_batch_normalize[CppTensor](x.wrapped[0])))

    class random:
        @staticmethod
        def bernoulli(shape, float p, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(op_random_bernoulli[CppTensor](normShape(shape).wrapped, p, device.wrapped[0])))

        @staticmethod
        def uniform(shape, float lower, float upper, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(op_random_uniform[CppTensor](normShape(shape).wrapped, lower, upper, device.wrapped[0])))

        @staticmethod
        def normal(shape, float mean, float sd, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(op_random_normal[CppTensor](normShape(shape).wrapped, mean, sd, device.wrapped[0])))

        @staticmethod
        def log_normal(shape, float mean, float sd, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(op_random_log_normal[CppTensor](normShape(shape).wrapped, mean, sd, device.wrapped[0])))

        @staticmethod
        def gumbel(shape, float mu, float beta, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(op_random_gumbel[CppTensor](normShape(shape).wrapped, mu, beta, device.wrapped[0])))

    @staticmethod
    def dropout(Tensor x, float rate, bool enabled):
        return Tensor.get_wrapper_with_new(new CppTensor(op_dropout(x.wrapped[0], rate, enabled)))
