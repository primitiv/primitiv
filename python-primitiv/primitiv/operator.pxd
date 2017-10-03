from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport Device
from primitiv.graph cimport Graph, Node
from primitiv.tensor cimport Tensor
from primitiv.shape cimport Shape
from primitiv.parameter cimport Parameter

cdef extern from "operator_template_wrapper.h" namespace "python_primitiv":

    Node Node_input_vector(const Shape &shape, const vector[float] &data, Device &dev, Graph &g) except +
    Node Node_input_vector(const Shape &shape, const vector[float] &data, Device &dev) except +
    Tensor Tensor_input_vector(const Shape &shape, const vector[float] &data, Device &dev) except +
    Node Node_input_parameter(Parameter &param, Graph &g) except +
    Node Node_input_parameter(Parameter &param) except +
    Tensor Tensor_input_parameter(Parameter &param) except +
    Node Node_copy(const Node &x, Device &dev) except +
    Tensor Tensor_copy(const Tensor &x, Device &dev) except +
    Node Node_pick(const Node &x, const vector[unsigned] &ids, unsigned dim) except +
    Tensor Tensor_pick(const Tensor &x, const vector[unsigned] &ids, unsigned dim) except +
    Node Node_slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) except +
    Tensor Tensor_slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper) except +
    Node Node_concat(const vector[Node] &xs, unsigned dim) except +
    Tensor Tensor_concat(const vector[Tensor] &xs, unsigned dim) except +
    Node Node_reshape(const Node &x, const Shape &new_shape) except +
    Tensor Tensor_reshape(const Tensor &x, const Shape &new_shape) except +
    Node Node_flatten(const Node &x) except +
    Tensor Tensor_flatten(const Tensor &x) except +
    Node Node_transpose(const Node &x) except +
    Tensor Tensor_transpose(const Tensor &x) except +
    Node Node_matmul(const Node &a, const Node &b) except +
    Tensor Tensor_matmul(const Tensor &a, const Tensor &b) except +
    Node Node_sqrt(const Node &x) except +
    Tensor Tensor_sqrt(const Tensor &x) except +
    Node Node_exp(const Node &x) except +
    Tensor Tensor_exp(const Tensor &x) except +
    Node Node_log(const Node &x) except +
    Tensor Tensor_log(const Tensor &x) except +
    Node Node_tanh(const Node &x) except +
    Tensor Tensor_tanh(const Tensor &x) except +
    Node Node_sigmoid(const Node &x) except +
    Tensor Tensor_sigmoid(const Tensor &x) except +
    Node Node_softplus(const Node &x) except +
    Tensor Tensor_softplus(const Tensor &x) except +
    Node Node_sin(const Node &x) except +
    Tensor Tensor_sin(const Tensor &x) except +
    Node Node_cos(const Node &x) except +
    Tensor Tensor_cos(const Tensor &x) except +
    Node Node_tan(const Node &x) except +
    Tensor Tensor_tan(const Tensor &x) except +
    Node Node_relu(const Node &x) except +
    Tensor Tensor_relu(const Tensor &x) except +
    Node Node_lrelu(const Node &x) except +
    Tensor Tensor_lrelu(const Tensor &x) except +
    Node Node_prelu(const Node &x, float a) except +
    Tensor Tensor_prelu(const Tensor &x, float a) except +
    Node Node_elu(const Node &x, float a) except +
    Tensor Tensor_elu(const Tensor &x, float a) except +
    Node Node_selu(const Node &x, float a, float s) except +
    Tensor Tensor_selu(const Tensor &x, float a, float s) except +
    Node Node_sum(const Node &x, unsigned dim) except +
    Tensor Tensor_sum(const Tensor &x, unsigned dim) except +
    Node Node_mean(const Node &x, unsigned dim) except +
    Tensor Node_mean(const Tensor &x, unsigned dim) except +
    Node Node_broadcast(const Node &x, unsigned dim, unsigned size) except +
    Tensor Tensor_broadcast(const Tensor &x, unsigned dim, unsigned size) except +
    Node Node_logsumexp(const Node &x, unsigned dim) except +
    Tensor Tensor_logsumexp(const Tensor &x, unsigned dim) except +
    Node Node_log_softmax(const Node &x, unsigned dim) except +
    Tensor Tensor_log_softmax(const Tensor &x, unsigned dim) except +
    Node Node_softmax(const Node &x, unsigned dim) except +
    Tensor Tensor_softmax(const Tensor &x, unsigned dim) except +
    Node Node_softmax_cross_entropy(const Node &x, const Node &t, unsigned dim) except +
    Tensor Tensor_softmax_cross_entropy(const Tensor &x, const Tensor &t, unsigned dim) except +
    Node Node_softmax_cross_entropy(const Node &x, const vector[unsigned] &ids, unsigned dim) except +
    Tensor Tensor_softmax_cross_entropy(const Tensor &x, const vector[unsigned] &ids, unsigned dim) except +

    Node Node_batch_sum(const Node &x) except +
    Tensor Tensor_batch_sum(const Tensor &x) except +
    Node Node_batch_mean(const Node &x) except +
    Tensor Tensor_batch_mean(const Tensor &x) except +
    Node Node_batch_normalize(const Node &x) except +
    Tensor Tensor_batch_normalize(const Tensor &x) except +

    Node Node_constant(const Shape &shape, float k, Device &dev, Graph &g) except +
    Node Node_zeros(const Shape &shape, Device &dev, Graph &g) except +
    Node Node_ones(const Shape &shape, Device &dev, Graph &g) except +
    Node Node_identity(unsigned size, Device &dev, Graph &g) except +
    Node Node_constant(const Shape &shape, float k, Device &dev) except +
    Tensor Tensor_constant(const Shape &shape, float k, Device &dev) except +
    Node Node_zeros(const Shape &shape, Device &dev) except +
    Tensor Tensor_zeros(const Shape &shape, Device &dev) except +
    Node Node_ones(const Shape &shape, Device &dev) except +
    Tensor Tensor_ones(const Shape &shape, Device &dev) except +
    Node Node_identity(unsigned size, Device &dev) except +
    Tensor Tensor_identity(unsigned size, Device &dev) except +

    Node Node_random_bernoulli(const Shape &shape, float p, Device &dev, Graph &g) except +
    Node Node_random_bernoulli(const Shape &shape, float p, Device &dev) except +
    Tensor Tensor_random_bernoulli(const Shape &shape, float p, Device &dev) except +
    Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev, Graph &g) except +
    Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev) except +
    Tensor Tensor_random_uniform(const Shape &shape, float lower, float upper, Device &dev) except +
    Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) except +
    Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev) except +
    Tensor Tensor_random_normal(const Shape &shape, float mean, float sd, Device &dev) except +
    Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) except +
    Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev) except +
    Tensor Tensor_random_log_normal(const Shape &shape, float mean, float sd, Device &dev) except +
    Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev, Graph &g) except +
    Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev) except +
    Tensor Tensor_random_gumbel(const Shape &shape, float mu, float beta, Device &dev) except +

    Node Node_dropout(const Node &x, float rate, bool enabled) except +
    Tensor Tensor_dropout(const Tensor &x, float rate, bool enabled) except +
