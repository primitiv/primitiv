#ifndef PYTHON_PRIMITIV_OP_TEMPLATE_WRAPPER_H_
#define PYTHON_PRIMITIV_OP_TEMPLATE_WRAPPER_H_

#include <primitiv/operators.h>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>

namespace python_primitiv {

using namespace primitiv;

inline Node Node_input_vector(const Shape &shape, const std::vector<float> &data, Device &dev, Graph &g) {
    return operators::input(shape, data, dev, g);
}

inline Node Node_input_vector(const Shape &shape, const std::vector<float> &data, Device &dev) {
    return operators::input<Node>(shape, data, dev);
}

inline Tensor Tensor_input_vector(const Shape &shape, const std::vector<float> &data, Device &dev) {
    return operators::input<Tensor>(shape, data, dev);
}

inline Node Node_parameter(Parameter &param, Graph &g) {
    return operators::parameter(param, g);
}

inline Node Node_parameter(Parameter &param) {
    return operators::parameter<Node>(param);
}

inline Tensor Tensor_parameter(Parameter &param) {
    return operators::parameter<Tensor>(param);
}

inline Node Node_copy(const Node &x, Device &dev) {
    return operators::copy<Node>(x, dev);
}

inline Tensor Tensor_copy(const Tensor &x, Device &dev) {
    return operators::copy<Tensor>(x, dev);
}

inline Node Node_pick(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::pick<Node>(x, ids, dim);
}

inline Tensor Tensor_pick(const Tensor &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::pick<Tensor>(x, ids, dim);
}

inline Node Node_slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) {
    return operators::slice<Node>(x, dim, lower, upper);
}

inline Tensor Tensor_slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
    return operators::slice<Tensor>(x, dim, lower, upper);
}

inline Node Node_concat(const std::vector<Node> &xs, unsigned dim) {
    return operators::concat<Node>(xs, dim);
}

inline Tensor Tensor_concat(const std::vector<Tensor> &xs, unsigned dim) {
    return operators::concat<Tensor>(xs, dim);
}

inline Node Node_reshape(const Node &x, const Shape &new_shape) {
    return operators::reshape<Node>(x, new_shape);
}

inline Tensor Tensor_reshape(const Tensor &x, const Shape &new_shape) {
    return operators::reshape<Tensor>(x, new_shape);
}

inline Node Node_flatten(const Node &x) {
    return operators::flatten<Node>(x);
}

inline Tensor Tensor_flatten(const Tensor &x) {
    return operators::flatten<Tensor>(x);
}

inline Node Node_transpose(const Node &x) {
    return operators::transpose<Node>(x);
}

inline Tensor Tensor_transpose(const Tensor &x) {
    return operators::transpose<Tensor>(x);
}

inline Node Node_matmul(const Node &a, const Node &b) {
    return operators::matmul<Node>(a, b);
}

inline Tensor Tensor_matmul(const Tensor &a, const Tensor &b) {
    return operators::matmul<Tensor>(a, b);
}

inline Node Node_sqrt(const Node &x) {
    return operators::sqrt<Node>(x);
}

inline Tensor Tensor_sqrt(const Tensor &x) {
    return operators::sqrt<Tensor>(x);
}

inline Node Node_exp(const Node &x) {
    return operators::exp<Node>(x);
}

inline Tensor Tensor_exp(const Tensor &x) {
    return operators::exp<Tensor>(x);
}

inline Node Node_log(const Node &x) {
    return operators::log<Node>(x);
}

inline Tensor Tensor_log(const Tensor &x) {
    return operators::log<Tensor>(x);
}

inline Node Node_tanh(const Node &x) {
    return operators::tanh<Node>(x);
}

inline Tensor Tensor_tanh(const Tensor &x) {
    return operators::tanh<Tensor>(x);
}

inline Node Node_sigmoid(const Node &x) {
    return operators::sigmoid<Node>(x);
}

inline Tensor Tensor_sigmoid(const Tensor &x) {
    return operators::sigmoid<Tensor>(x);
}

inline Node Node_softplus(const Node &x) {
    return operators::softplus<Node>(x);
}

inline Tensor Tensor_softplus(const Tensor &x) {
    return operators::softplus<Tensor>(x);
}

inline Node Node_sin(const Node &x) {
    return operators::sin<Node>(x);
}

inline Tensor Tensor_sin(const Tensor &x) {
    return operators::sin<Tensor>(x);
}

inline Node Node_cos(const Node &x) {
    return operators::cos<Node>(x);
}

inline Tensor Tensor_cos(const Tensor &x) {
    return operators::cos<Tensor>(x);
}

inline Node Node_tan(const Node &x) {
    return operators::tan<Node>(x);
}

inline Tensor Tensor_tan(const Tensor &x) {
    return operators::tan<Tensor>(x);
}

inline Node Node_relu(const Node &x) {
    return operators::relu<Node>(x);
}

inline Tensor Tensor_relu(const Tensor &x) {
    return operators::relu<Tensor>(x);
}

inline Node Node_lrelu(const Node &x) {
    return operators::lrelu<Node>(x);
}

inline Tensor Tensor_lrelu(const Tensor &x) {
    return operators::lrelu<Tensor>(x);
}

inline Node Node_prelu(const Node &x, float a) {
    return operators::prelu<Node>(x, a);
}

inline Tensor Tensor_prelu(const Tensor &x, float a) {
    return operators::prelu<Tensor>(x, a);
}

inline Node Node_elu(const Node &x, float a) {
    return operators::elu<Node>(x, a);
}

inline Tensor Tensor_elu(const Tensor &x, float a) {
    return operators::elu<Tensor>(x, a);
}

inline Node Node_selu(const Node &x, float a, float s) {
    return operators::selu<Node>(x, a, s);
}

inline Tensor Tensor_selu(const Tensor &x, float a, float s) {
    return operators::selu<Tensor>(x, a, s);
}

inline Node Node_sum(const Node &x, unsigned dim) {
    return operators::sum<Node>(x, dim);
}

inline Node Node_sum_container(vector<Node> &xs) {
    return operators::sum<vector<Node>>(xs);
}

inline Tensor Tensor_sum(const Tensor &x, unsigned dim) {
    return operators::sum<Tensor>(x, dim);
}

inline Node Node_mean(const Node &x, unsigned dim) {
    return operators::mean<Node>(x, dim);
}

inline Tensor Node_mean(const Tensor &x, unsigned dim) {
    return operators::mean<Tensor>(x, dim);
}

inline Node Node_broadcast(const Node &x, unsigned dim, unsigned size) {
    return operators::broadcast<Node>(x, dim, size);
}

inline Tensor Tensor_broadcast(const Tensor &x, unsigned dim, unsigned size) {
    return operators::broadcast<Tensor>(x, dim, size);
}

inline Node Node_logsumexp(const Node &x, unsigned dim) {
    return operators::logsumexp<Node>(x, dim);
}

inline Tensor Tensor_logsumexp(const Tensor &x, unsigned dim) {
    return operators::logsumexp<Tensor>(x, dim);
}

inline Node Node_log_softmax(const Node &x, unsigned dim) {
    return operators::log_softmax<Node>(x, dim);
}

inline Tensor Tensor_log_softmax(const Tensor &x, unsigned dim) {
    return operators::log_softmax<Tensor>(x, dim);
}

inline Node Node_softmax(const Node &x, unsigned dim) {
    return operators::softmax<Node>(x, dim);
}

inline Tensor Tensor_softmax(const Tensor &x, unsigned dim) {
    return operators::softmax<Tensor>(x, dim);
}

inline Node Node_softmax_cross_entropy(const Node &x, const Node &t, unsigned dim) {
    return operators::softmax_cross_entropy<Node>(x, t, dim);
}

inline Tensor Tensor_softmax_cross_entropy(const Tensor &x, const Tensor &t, unsigned dim) {
    return operators::softmax_cross_entropy<Tensor>(x, t, dim);
}

inline Node Node_softmax_cross_entropy(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::softmax_cross_entropy<Node>(x, ids, dim);
}

inline Tensor Tensor_softmax_cross_entropy(const Tensor &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::softmax_cross_entropy<Tensor>(x, ids, dim);
}

inline Node Node_batch_sum(const Node &x) {
    return operators::batch::sum<Node>(x);
}

inline Tensor Tensor_batch_sum(const Tensor &x) {
    return operators::batch::sum<Tensor>(x);
}

inline Node Node_batch_mean(const Node &x) {
    return operators::batch::mean<Node>(x);
}

inline Tensor Tensor_batch_mean(const Tensor &x) {
    return operators::batch::mean<Tensor>(x);
}

inline Node Node_batch_normalize(const Node &x) {
    return operators::batch::normalize<Node>(x);
}

inline Tensor Tensor_batch_normalize(const Tensor &x) {
    return operators::batch::normalize<Tensor>(x);
}

inline Node Node_constant(const Shape &shape, float k, Device &dev, Graph &g) {
    return operators::constant(shape, k, dev, g);
}

inline Node Node_zeros(const Shape &shape, Device &dev, Graph &g) {
    return operators::zeros(shape, dev, g);
}

inline Node Node_ones(const Shape &shape, Device &dev, Graph &g) {
    return operators::ones(shape, dev, g);
}

inline Node Node_identity(unsigned size, Device &dev, Graph &g) {
    return operators::identity(size, dev, g);
}

inline Node Node_constant(const Shape &shape, float k, Device &dev) {
    return operators::constant<Node>(shape, k, dev);
}

inline Tensor Tensor_constant(const Shape &shape, float k, Device &dev) {
    return operators::constant<Tensor>(shape, k, dev);
}

inline Node Node_zeros(const Shape &shape, Device &dev) {
    return operators::zeros<Node>(shape, dev);
}

inline Tensor Tensor_zeros(const Shape &shape, Device &dev) {
    return operators::zeros<Tensor>(shape, dev);
}

inline Node Node_ones(const Shape &shape, Device &dev) {
    return operators::ones<Node>(shape, dev);
}

inline Tensor Tensor_ones(const Shape &shape, Device &dev) {
    return operators::ones<Tensor>(shape, dev);
}

inline Node Node_identity(unsigned size, Device &dev) {
    return operators::identity<Node>(size, dev);
}

inline Tensor Tensor_identity(unsigned size, Device &dev) {
    return operators::identity<Tensor>(size, dev);
}

inline Node Node_random_bernoulli(const Shape &shape, float p, Device &dev, Graph &g) {
    return operators::random::bernoulli(shape, p, dev, g);
}

inline Node Node_random_bernoulli(const Shape &shape, float p, Device &dev) {
    return operators::random::bernoulli<Node>(shape, p, dev);
}

inline Tensor Tensor_random_bernoulli(const Shape &shape, float p, Device &dev) {
    return operators::random::bernoulli<Tensor>(shape, p, dev);
}

inline Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev, Graph &g) {
    return operators::random::uniform(shape, lower, upper, dev, g);
}

inline Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev) {
    return operators::random::uniform<Node>(shape, lower, upper, dev);
}

inline Tensor Tensor_random_uniform(const Shape &shape, float lower, float upper, Device &dev) {
    return operators::random::uniform<Tensor>(shape, lower, upper, dev);
}

inline Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
    return operators::random::normal(shape, mean, sd, dev, g);
}

inline Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::normal<Node>(shape, mean, sd, dev);
}

inline Tensor Tensor_random_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::normal<Tensor>(shape, mean, sd, dev);
}

inline Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
    return operators::random::log_normal(shape, mean, sd, dev, g);
}

inline Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::log_normal<Node>(shape, mean, sd, dev);
}

inline Tensor Tensor_random_log_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::log_normal<Tensor>(shape, mean, sd, dev);
}

inline Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev, Graph &g) {
    return operators::random::gumbel(shape, mu, beta, dev, g);
}

inline Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev) {
    return operators::random::gumbel<Node>(shape, mu, beta, dev);
}

inline Tensor Tensor_random_gumbel(const Shape &shape, float mu, float beta, Device &dev) {
    return operators::random::gumbel<Tensor>(shape, mu, beta, dev);
}

inline Node Node_dropout(const Node &x, float rate, bool enabled) {
    return operators::dropout<Node>(x, rate, enabled);
}

inline Tensor Tensor_dropout(const Tensor &x, float rate, bool enabled) {
    return operators::dropout<Tensor>(x, rate, enabled);
}

}

#endif
