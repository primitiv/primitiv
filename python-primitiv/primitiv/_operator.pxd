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
    CppNode Node_sum(const CppNode &x, unsigned dim) except +
    CppNode Node_sum_container(const vector[CppNode] &xs) except +
    CppNode Node_mean(const CppNode &x, unsigned dim) except +
    CppNode Node_mean_container(const vector[CppNode] &xs) except +

    CppTensor Tensor_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev) except +
    CppTensor Tensor_parameter(CppParameter &param) except +
    CppTensor Tensor_sum(const CppTensor &x, unsigned dim) except +
    CppTensor Tensor_sum_container(const vector[CppTensor] &xs) except +
    CppTensor Tensor_mean(const CppTensor &x, unsigned dim) except +
    CppTensor Tensor_mean_container(const vector[CppTensor] &xs) except +


cdef extern from "primitiv/operators.h":
    Var op_copy "primitiv::operators::copy" [Var](const Var &x, CppDevice &dev) except +
    Var op_pick "primitiv::operators::pick" [Var](const Var &x, const vector[unsigned] &ids, unsigned dim) except +
    Var op_slice "primitiv::operators::slice" [Var](const Var &x, unsigned dim, unsigned lower, unsigned upper) except +
    Var op_concat "primitiv::operators::concat" [Var](const vector[Var] &xs, unsigned dim) except +
    Var op_reshape "primitiv::operators::reshape" [Var](const Var &x, const CppShape &new_shape) except +
    Var op_flatten "primitiv::operators::flatten" [Var](const Var &x) except +
    Var op_transpose "primitiv::operators::transpose" [Var](const Var &x) except +
    Var op_matmul "primitiv::operators::matmul" [Var](const Var &a, const Var &b) except +
    Var op_sqrt "primitiv::operators::sqrt" [Var](const Var &x) except +
    Var op_exp "primitiv::operators::exp" [Var](const Var &x) except +
    Var op_log "primitiv::operators::log" [Var](const Var &x) except +
    Var op_ipow "primitiv::operators::ipow" [Var](const Var &x, int k) except +
    Var op_pow "primitiv::operators::pow" [Var](const Var &x, float k) except +
    Var op_pow "primitiv::operators::pow" [Var](float x, const Var &k) except +
    Var op_pow "primitiv::operators::pow" [Var](const Var &x, const Var &k) except +
    Var op_tanh "primitiv::operators::tanh" [Var](const Var &x) except +
    Var op_sigmoid "primitiv::operators::sigmoid" [Var](const Var &x) except +
    Var op_softplus "primitiv::operators::softplus" [Var](const Var &x) except +
    Var op_sin "primitiv::operators::sin" [Var](const Var &x) except +
    Var op_cos "primitiv::operators::cos" [Var](const Var &x) except +
    Var op_tan "primitiv::operators::tan" [Var](const Var &x) except +
    Var op_relu "primitiv::operators::relu" [Var](const Var &x) except +
    Var op_lrelu "primitiv::operators::lrelu" [Var](const Var &x) except +
    Var op_prelu "primitiv::operators::prelu" [Var](const Var &x, float a) except +
    Var op_elu "primitiv::operators::elu" [Var](const Var &x, float a) except +
    Var op_selu "primitiv::operators::selu" [Var](const Var &x, float a, float s) except +
    Var op_broadcast "primitiv::operators::broadcast" [Var](const Var &x, unsigned dim, unsigned size) except +
    Var op_logsumexp "primitiv::operators::logsumexp" [Var](const Var &x, unsigned dim) except +
    Var op_log_softmax "primitiv::operators::log_softmax" [Var](const Var &x, unsigned dim) except +
    Var op_softmax "primitiv::operators::softmax" [Var](const Var &x, unsigned dim) except +
    Var op_softmax_cross_entropy "primitiv::operators::softmax_cross_entropy" [Var](const Var &x, const Var &t, unsigned dim) except +
    Var op_softmax_cross_entropy "primitiv::operators::softmax_cross_entropy" [Var](const Var &x, const vector[unsigned] &ids, unsigned dim) except +
    CppNode op_constant "primitiv::operators::constant" (const CppShape &shape, float k, CppDevice &dev, CppGraph &g) except +
    CppNode op_zeros "primitiv::operators::zeros" (const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode op_ones "primitiv::operators::ones" (const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode op_identity "primitiv::operators::identity" (unsigned size, CppDevice &dev, CppGraph &g) except +
    Var op_constant "primitiv::operators::constant" [Var](const CppShape &shape, float k, CppDevice &dev) except +
    Var op_zeros "primitiv::operators::zeros" [Var](const CppShape &shape, CppDevice &dev) except +
    Var op_ones "primitiv::operators::ones" [Var](const CppShape &shape, CppDevice &dev) except +
    Var op_identity "primitiv::operators::identity" [Var](unsigned size, CppDevice &dev) except +
    Var op_dropout "primitiv::operators::dropout" [Var](const Var &x, float rate, bool enabled) except +


cdef extern from "primitiv/operators.h":
    Var op_batch_sum "primitiv::operators::batch::sum" [Var](const Var &x) except +
    Var op_batch_mean "primitiv::operators::batch::mean" [Var](const Var &x) except +
    Var op_batch_normalize "primitiv::operators::batch::normalize" [Var](const Var &x) except +


cdef extern from "primitiv/operators.h":

    CppNode op_random_bernoulli "primitiv::operators::random::bernoulli" (const CppShape &shape, float p, CppDevice &dev, CppGraph &g) except +
    Var op_random_bernoulli "primitiv::operators::random::bernoulli" [Var](const CppShape &shape, float p, CppDevice &dev) except +
    CppNode op_random_uniform "primitiv::operators::random::uniform" (const CppShape &shape, float lower, float upper, CppDevice &dev, CppGraph &g) except +
    Var op_random_uniform "primitiv::operators::random::uniform" [Var](const CppShape &shape, float lower, float upper, CppDevice &dev) except +
    CppNode op_random_normal "primitiv::operators::random::normal" (const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    Var op_random_normal "primitiv::operators::random::normal" [Var](const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode op_random_log_normal "primitiv::operators::random::log_normal" (const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    Var op_random_log_normal "primitiv::operators::random::log_normal" [Var](const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode op_random_gumbel "primitiv::operators::random::gumbel" (const CppShape &shape, float mu, float beta, CppDevice &dev, CppGraph &g) except +
    Var op_random_gumbel "primitiv::operators::random::gumbel" [Var](const CppShape &shape, float mu, float beta, CppDevice &dev) except +
