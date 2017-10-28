from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport _Device
from primitiv._shape cimport _Shape, normShape
from primitiv._tensor cimport _Tensor, CppTensor
from primitiv._graph cimport _Graph, wrapNode, CppNode, _Node
from primitiv._parameter cimport _Parameter

from utils cimport ndarrays_to_vector

cimport numpy as np
import numpy as np


class _operators:

    @staticmethod
    def input(data, shape = None, _Device device = None, _Graph g = None):
        cdef vector[float] data_vector
        if device is None:
            device = _Device.get_default()
        if isinstance(data, np.ndarray):
            data = [data]
        elif not isinstance(data, list):
            raise TypeError("Argument 'data' has incorrect type (list or numpy.ndarray)")
        if len(data) == 0:
            raise TypeError("data is a list, but it contains no item")
        if isinstance(data[0], (float, int)):
            if shape is None:
                raise TypeError("shape is required when data contains scalars")
            data_vector = <vector[float]> data
        else:
            if shape is None:
                shape = _Shape(data[0].shape, len(data))
            data_vector = ndarrays_to_vector(data)
        if g is not None:
            return wrapNode(Node_input_vector(normShape(shape).wrapped, data_vector, device.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_input_vector(normShape(shape).wrapped, data_vector, device.wrapped[0]))

    @staticmethod
    def parameter(_Parameter param, _Graph g = None):
        if g is not None:
            return wrapNode(Node_parameter(param.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_parameter(param.wrapped[0]))

    @staticmethod
    def copy(_Node x, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return wrapNode(op_copy(x.wrapped, device.wrapped[0]))

    @staticmethod
    def pick(_Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(op_pick(x.wrapped, ids, dim))

    @staticmethod
    def slice(_Node x, unsigned dim, unsigned lower, unsigned upper):
        return wrapNode(op_slice(x.wrapped, dim, lower, upper))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppNode] vec
        cdef _Node x
        for x in xs:
            vec.push_back(x.wrapped)
        return wrapNode(op_concat(vec, dim))

    @staticmethod
    def reshape(_Node x, _Shape new_shape):
        return wrapNode(op_reshape(x.wrapped, new_shape.wrapped))

    @staticmethod
    def flatten(_Node x):
        return wrapNode(op_flatten(x.wrapped))

    @staticmethod
    def transpose(_Node x):
        return wrapNode(op_transpose(x.wrapped))

    @staticmethod
    def matmul(_Node a, _Node b):
        return wrapNode(op_matmul(a.wrapped, b.wrapped))

    @staticmethod
    def sqrt(_Node x):
        return wrapNode(op_sqrt(x.wrapped))

    @staticmethod
    def exp(_Node x):
        return wrapNode(op_exp(x.wrapped))

    @staticmethod
    def log(_Node x):
        return wrapNode(op_log(x.wrapped))

    @staticmethod
    def pow(x, k):
        if isinstance(x, _Node) and isinstance(k, int):
            return wrapNode(op_pow((<_Node> x).wrapped, <unsigned> k))
        if isinstance(x, _Node) and isinstance(k, float):
            return wrapNode(op_pow((<_Node> x).wrapped, <float> k))
        elif isinstance(x, (int, float)) and isinstance(k, _Node):
            return wrapNode(op_pow(<float> x, (<_Node> k).wrapped))
        elif isinstance(x, _Node) and isinstance(k, _Node):
            return wrapNode(op_pow((<_Node> x).wrapped, (<_Node> k).wrapped))
        else:
            raise TypeError("Argument 'x' or 'k' has incorrect type (Node or float or int)")

    @staticmethod
    def tanh(_Node x):
        return wrapNode(op_tanh(x.wrapped))

    @staticmethod
    def sigmoid(_Node x):
        return wrapNode(op_sigmoid(x.wrapped))

    @staticmethod
    def softplus(_Node x):
        return wrapNode(op_softplus(x.wrapped))

    @staticmethod
    def sin(_Node x):
        return wrapNode(op_sin(x.wrapped))

    @staticmethod
    def cos(_Node x):
        return wrapNode(op_cos(x.wrapped))

    @staticmethod
    def tan(_Node x):
        return wrapNode(op_tan(x.wrapped))

    @staticmethod
    def relu(_Node x):
        return wrapNode(op_relu(x.wrapped))

    @staticmethod
    def lrelu(_Node x):
        return wrapNode(op_lrelu(x.wrapped))

    @staticmethod
    def prelu(_Node x, float a):
        return wrapNode(op_prelu(x.wrapped, a))

    @staticmethod
    def elu(_Node x, float a):
        return wrapNode(op_elu(x.wrapped, a))

    @staticmethod
    def selu(_Node x, float a, float s):
        return wrapNode(op_selu(x.wrapped, a, s))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppNode] xs
        cdef _Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(Node_sum_container(xs))
        else:
            return wrapNode(Node_sum((<_Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppNode] xs
        cdef _Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(Node_mean_container(xs))
        else:
            return wrapNode(Node_mean((<_Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def broadcast(_Node x, unsigned dim, unsigned size):
        return wrapNode(op_broadcast(x.wrapped, dim, size))

    @staticmethod
    def logsumexp(_Node x, unsigned dim):
        return wrapNode(op_logsumexp(x.wrapped, dim))

    @staticmethod
    def log_softmax(_Node x, unsigned dim):
        return wrapNode(op_log_softmax(x.wrapped, dim))

    @staticmethod
    def softmax(_Node x, unsigned dim):
        return wrapNode(op_softmax(x.wrapped, dim))

    @staticmethod
    def softmax_cross_entropy(_Node x, t, unsigned dim):
        if isinstance(t, _Node):
            return wrapNode(op_softmax_cross_entropy(x.wrapped, (<_Node> t).wrapped, dim))
        elif isinstance(t, list):
            return wrapNode(op_softmax_cross_entropy(x.wrapped, <vector[unsigned]> t, dim))
        else:
            raise TypeError("Argument 't' has incorrect type (list or Node)")

    @staticmethod
    def constant(shape, float k, _Device device = None, _Graph g = None):
        if device is None:
            device = _Device.get_default()
        if g is None:
            return wrapNode(op_constant[CppNode](normShape(shape).wrapped, k, device.wrapped[0]))
        else:
            return wrapNode(op_constant(normShape(shape).wrapped, k, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def zeros(shape, _Device device = None, _Graph g = None):
        if device is None:
            device = _Device.get_default()
        if g is None:
            return wrapNode(op_zeros[CppNode](normShape(shape).wrapped, device.wrapped[0]))
        else:
            return wrapNode(op_zeros(normShape(shape).wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def ones(shape, _Device device = None, _Graph g = None):
        if device is None:
            device = _Device.get_default()
        if g is None:
            return wrapNode(op_ones[CppNode](normShape(shape).wrapped, device.wrapped[0]))
        else:
            return wrapNode(op_ones(normShape(shape).wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def identity(unsigned size, _Device device = None, _Graph g = None):
        if device is None:
            device = _Device.get_default()
        if g is None:
            return wrapNode(op_identity[CppNode](size, device.wrapped[0]))
        else:
            return wrapNode(op_identity(size, device.wrapped[0], g.wrapped[0]))

    class batch:
        @staticmethod
        def sum(_Node x):
            return wrapNode(op_batch_sum[CppNode](x.wrapped))

        @staticmethod
        def mean(_Node x):
            return wrapNode(op_batch_mean[CppNode](x.wrapped))

        @staticmethod
        def normalize(_Node x):
            return wrapNode(op_batch_normalize[CppNode](x.wrapped))

    class random:
        @staticmethod
        def bernoulli(shape, float p, _Device device = None, _Graph g = None):
            if device is None:
                device = _Device.get_default()
            if g is None:
                return wrapNode(op_random_bernoulli[CppNode](normShape(shape).wrapped, p, device.wrapped[0]))
            else:
                return wrapNode(op_random_bernoulli(normShape(shape).wrapped, p, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def uniform(shape, float lower, float upper, _Device device = None, _Graph g = None):
            if device is None:
                device = _Device.get_default()
            if g is None:
                return wrapNode(op_random_uniform[CppNode](normShape(shape).wrapped, lower, upper, device.wrapped[0]))
            else:
                return wrapNode(op_random_uniform(normShape(shape).wrapped, lower, upper, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def normal(shape, float mean, float sd, _Device device = None, _Graph g = None):
            if device is None:
                device = _Device.get_default()
            if g is None:
                return wrapNode(op_random_normal[CppNode](normShape(shape).wrapped, mean, sd, device.wrapped[0]))
            else:
                return wrapNode(op_random_normal(normShape(shape).wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def log_normal(shape, float mean, float sd, _Device device = None, _Graph g = None):
            if device is None:
                device = _Device.get_default()
            if g is None:
                return wrapNode(op_random_log_normal[CppNode](normShape(shape).wrapped, mean, sd, device.wrapped[0]))
            else:
                return wrapNode(op_random_log_normal(normShape(shape).wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def gumbel(shape, float mu, float beta, _Device device = None, _Graph g = None):
            if device is None:
                device = _Device.get_default()
            if g is None:
                return wrapNode(op_random_gumbel[CppNode](normShape(shape).wrapped, mu, beta, device.wrapped[0]))
            else:
                return wrapNode(op_random_gumbel(normShape(shape).wrapped, mu, beta, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def dropout(_Node x, float rate, bool enabled):
        return wrapNode(op_dropout(x.wrapped, rate, enabled))


class _tensor_operators:

    @staticmethod
    def input(data, shape = None, _Device device = None):
        cdef vector[float] data_vector
        if device is None:
            device = _Device.get_default()
        if isinstance(data, np.ndarray):
            data = [data]
        elif not isinstance(data, list):
            raise TypeError("Argument 'data' has incorrect type (list or numpy.ndarray)")
        if len(data) == 0:
            raise TypeError("data is a list, but it contains no item")
        if isinstance(data[0], (float, int)):
            if shape is None:
                raise TypeError("shape is required when data contains scalars")
            data_vector = <vector[float]> data
        else:
            if shape is None:
                shape = _Shape(data[0].shape, len(data))
            data_vector = ndarrays_to_vector(data)
        return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_input_vector(normShape(shape).wrapped, data_vector, device.wrapped[0])))

    @staticmethod
    def parameter(_Parameter param):
        return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_parameter(param.wrapped[0])))

    @staticmethod
    def copy(_Tensor x, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Tensor.get_wrapper_with_new(new CppTensor(op_copy(x.wrapped[0], device.wrapped[0])))

    @staticmethod
    def pick(_Tensor x, vector[unsigned] ids, unsigned dim):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_pick(x.wrapped[0], ids, dim)))

    @staticmethod
    def slice(_Tensor x, unsigned dim, unsigned lower, unsigned upper):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_slice(x.wrapped[0], dim, lower, upper)))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppTensor] vec
        cdef _Tensor x
        for x in xs:
            vec.push_back(x.wrapped[0])
        return _Tensor.get_wrapper_with_new(new CppTensor(op_concat(vec, dim)))

    @staticmethod
    def reshape(_Tensor x, _Shape new_shape):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_reshape(x.wrapped[0], new_shape.wrapped)))

    @staticmethod
    def flatten(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_flatten(x.wrapped[0])))

    @staticmethod
    def transpose(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_transpose(x.wrapped[0])))

    @staticmethod
    def matmul(_Tensor a, _Tensor b):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_matmul(a.wrapped[0], b.wrapped[0])))

    @staticmethod
    def sqrt(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_sqrt(x.wrapped[0])))

    @staticmethod
    def exp(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_exp(x.wrapped[0])))

    @staticmethod
    def log(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_log(x.wrapped[0])))

    @staticmethod
    def pow(x, k):
        if isinstance(x, _Tensor) and isinstance(k, int) and k >= 0:
            return _Tensor.get_wrapper_with_new(new CppTensor(op_pow((<_Tensor> x).wrapped[0], <unsigned> k)))
        elif isinstance(x, _Tensor) and isinstance(k, (int, (int, float))):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_pow((<_Tensor> x).wrapped[0], <float> k)))
        elif isinstance(x, (int, float)) and isinstance(k, _Tensor):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_pow(<float> x, (<_Tensor> k).wrapped[0])))
        elif isinstance(x, _Tensor) and isinstance(k, _Tensor):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_pow((<_Tensor> x).wrapped[0], (<_Tensor> k).wrapped[0])))
        else:
            raise TypeError("Argument 'x' or 'k' has incorrect type (Node or float or int)")

    @staticmethod
    def tanh(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_tanh(x.wrapped[0])))

    @staticmethod
    def sigmoid(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_sigmoid(x.wrapped[0])))

    @staticmethod
    def softplus(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_softplus(x.wrapped[0])))

    @staticmethod
    def sin(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_sin(x.wrapped[0])))

    @staticmethod
    def cos(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_cos(x.wrapped[0])))

    @staticmethod
    def tan(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_tan(x.wrapped[0])))

    @staticmethod
    def relu(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_relu(x.wrapped[0])))

    @staticmethod
    def lrelu(_Tensor x):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_lrelu(x.wrapped[0])))

    @staticmethod
    def prelu(_Tensor x, float a):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_prelu(x.wrapped[0], a)))

    @staticmethod
    def elu(_Tensor x, float a):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_elu(x.wrapped[0], a)))

    @staticmethod
    def selu(_Tensor x, float a, float s):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_selu(x.wrapped[0], a, s)))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppTensor] xs
        cdef _Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_sum_container(xs)))
        else:
            return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_sum((<_Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppTensor] xs
        cdef _Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_mean_container(xs)))
        else:
            return _Tensor.get_wrapper_with_new(new CppTensor(Tensor_mean((<_Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def broadcast(_Tensor x, unsigned dim, unsigned size):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_broadcast(x.wrapped[0], dim, size)))

    @staticmethod
    def logsumexp(_Tensor x, unsigned dim):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_logsumexp(x.wrapped[0], dim)))

    @staticmethod
    def log_softmax(_Tensor x, unsigned dim):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_log_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax(_Tensor x, unsigned dim):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax_cross_entropy(_Tensor x, t, unsigned dim):
        if isinstance(t, _Tensor):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_softmax_cross_entropy(x.wrapped[0], (<_Tensor> t).wrapped[0], dim)))
        elif isinstance(t, list):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_softmax_cross_entropy(x.wrapped[0], <vector[unsigned]> t, dim)))
        else:
            raise TypeError("Argument 't' has incorrect type (list or Node)")

    @staticmethod
    def constant(shape, float k, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Tensor.get_wrapper_with_new(new CppTensor(op_constant[CppTensor](normShape(shape).wrapped, k, device.wrapped[0])))

    @staticmethod
    def zeros(shape, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Tensor.get_wrapper_with_new(new CppTensor(op_zeros[CppTensor](normShape(shape).wrapped, device.wrapped[0])))

    @staticmethod
    def ones(shape, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Tensor.get_wrapper_with_new(new CppTensor(op_ones[CppTensor](normShape(shape).wrapped, device.wrapped[0])))

    @staticmethod
    def identity(unsigned size, _Device device = None):
        if device is None:
            device = _Device.get_default()
        return _Tensor.get_wrapper_with_new(new CppTensor(op_identity[CppTensor](size, device.wrapped[0])))

    class batch:
        @staticmethod
        def sum(_Tensor x):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_batch_sum[CppTensor](x.wrapped[0])))

        @staticmethod
        def mean(_Tensor x):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_batch_mean[CppTensor](x.wrapped[0])))

        @staticmethod
        def normalize(_Tensor x):
            return _Tensor.get_wrapper_with_new(new CppTensor(op_batch_normalize[CppTensor](x.wrapped[0])))

    class random:
        @staticmethod
        def bernoulli(shape, float p, _Device device = None):
            if device is None:
                device = _Device.get_default()
            return _Tensor.get_wrapper_with_new(new CppTensor(op_random_bernoulli[CppTensor](normShape(shape).wrapped, p, device.wrapped[0])))

        @staticmethod
        def uniform(shape, float lower, float upper, _Device device = None):
            if device is None:
                device = _Device.get_default()
            return _Tensor.get_wrapper_with_new(new CppTensor(op_random_uniform[CppTensor](normShape(shape).wrapped, lower, upper, device.wrapped[0])))

        @staticmethod
        def normal(shape, float mean, float sd, _Device device = None):
            if device is None:
                device = _Device.get_default()
            return _Tensor.get_wrapper_with_new(new CppTensor(op_random_normal[CppTensor](normShape(shape).wrapped, mean, sd, device.wrapped[0])))

        @staticmethod
        def log_normal(shape, float mean, float sd, _Device device = None):
            if device is None:
                device = _Device.get_default()
            return _Tensor.get_wrapper_with_new(new CppTensor(op_random_log_normal[CppTensor](normShape(shape).wrapped, mean, sd, device.wrapped[0])))

        @staticmethod
        def gumbel(shape, float mu, float beta, _Device device = None):
            if device is None:
                device = _Device.get_default()
            return _Tensor.get_wrapper_with_new(new CppTensor(op_random_gumbel[CppTensor](normShape(shape).wrapped, mu, beta, device.wrapped[0])))

    @staticmethod
    def dropout(_Tensor x, float rate, bool enabled):
        return _Tensor.get_wrapper_with_new(new CppTensor(op_dropout(x.wrapped[0], rate, enabled)))
