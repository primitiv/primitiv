from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._graph cimport CppGraph, CppNode
from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape
from primitiv._parameter cimport CppParameter

cdef extern from "operator_template_wrapper.h" namespace "python_primitiv":

    CppNode Node_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev, CppGraph &g) except +
    CppNode Node_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev) except +
    CppNode Node_parameter(CppParameter &param, CppGraph &g) except +
    CppNode Node_parameter(CppParameter &param) except +
    CppNode Node_copy(const CppNode &x, CppDevice &dev) except +
    CppNode Node_pick(const CppNode &x, const vector[unsigned] &ids, unsigned dim) except +
    CppNode Node_slice(const CppNode &x, unsigned dim, unsigned lower, unsigned upper) except +
    CppNode Node_concat(const vector[CppNode] &xs, unsigned dim) except +
    CppNode Node_reshape(const CppNode &x, const CppShape &new_shape) except +
    CppNode Node_flatten(const CppNode &x) except +
    CppNode Node_transpose(const CppNode &x) except +
    CppNode Node_matmul(const CppNode &a, const CppNode &b) except +
    CppNode Node_sqrt(const CppNode &x) except +
    CppNode Node_exp(const CppNode &x) except +
    CppNode Node_log(const CppNode &x) except +
    CppNode Node_pow(const CppNode &x, float k) except +
    CppNode Node_pow(float x, const CppNode &k) except +
    CppNode Node_pow(const CppNode &x, const CppNode &k) except +
    CppNode Node_tanh(const CppNode &x) except +
    CppNode Node_sigmoid(const CppNode &x) except +
    CppNode Node_softplus(const CppNode &x) except +
    CppNode Node_sin(const CppNode &x) except +
    CppNode Node_cos(const CppNode &x) except +
    CppNode Node_tan(const CppNode &x) except +
    CppNode Node_relu(const CppNode &x) except +
    CppNode Node_lrelu(const CppNode &x) except +
    CppNode Node_prelu(const CppNode &x, float a) except +
    CppNode Node_elu(const CppNode &x, float a) except +
    CppNode Node_selu(const CppNode &x, float a, float s) except +
    CppNode Node_sum(const CppNode &x, unsigned dim) except +
    CppNode Node_sum_container(const vector[CppNode] &xs) except +
    CppNode Node_mean(const CppNode &x, unsigned dim) except +
    CppNode Node_mean_container(const vector[CppNode] &xs) except +
    CppNode Node_broadcast(const CppNode &x, unsigned dim, unsigned size) except +
    CppNode Node_logsumexp(const CppNode &x, unsigned dim) except +
    CppNode Node_log_softmax(const CppNode &x, unsigned dim) except +
    CppNode Node_softmax(const CppNode &x, unsigned dim) except +
    CppNode Node_softmax_cross_entropy(const CppNode &x, const CppNode &t, unsigned dim) except +
    CppNode Node_softmax_cross_entropy(const CppNode &x, const vector[unsigned] &ids, unsigned dim) except +
    CppNode Node_batch_sum(const CppNode &x) except +
    CppNode Node_batch_mean(const CppNode &x) except +
    CppNode Node_batch_normalize(const CppNode &x) except +
    CppNode Node_constant(const CppShape &shape, float k, CppDevice &dev, CppGraph &g) except +
    CppNode Node_zeros(const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode Node_ones(const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode Node_identity(unsigned size, CppDevice &dev, CppGraph &g) except +
    CppNode Node_constant(const CppShape &shape, float k, CppDevice &dev) except +
    CppNode Node_zeros(const CppShape &shape, CppDevice &dev) except +
    CppNode Node_ones(const CppShape &shape, CppDevice &dev) except +
    CppNode Node_identity(unsigned size, CppDevice &dev) except +
    CppNode Node_random_bernoulli(const CppShape &shape, float p, CppDevice &dev, CppGraph &g) except +
    CppNode Node_random_bernoulli(const CppShape &shape, float p, CppDevice &dev) except +
    CppNode Node_random_uniform(const CppShape &shape, float lower, float upper, CppDevice &dev, CppGraph &g) except +
    CppNode Node_random_uniform(const CppShape &shape, float lower, float upper, CppDevice &dev) except +
    CppNode Node_random_normal(const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    CppNode Node_random_normal(const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode Node_random_log_normal(const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    CppNode Node_random_log_normal(const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode Node_random_gumbel(const CppShape &shape, float mu, float beta, CppDevice &dev, CppGraph &g) except +
    CppNode Node_random_gumbel(const CppShape &shape, float mu, float beta, CppDevice &dev) except +
    CppNode Node_dropout(const CppNode &x, float rate, bool enabled) except +
