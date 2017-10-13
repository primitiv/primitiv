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

inline Node Node_parameter(Parameter &param, Graph &g) {
    return operators::parameter(param, g);
}

inline Node Node_parameter(Parameter &param) {
    return operators::parameter<Node>(param);
}

inline Node Node_copy(const Node &x, Device &dev) {
    return operators::copy<Node>(x, dev);
}

inline Node Node_pick(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::pick<Node>(x, ids, dim);
}

inline Node Node_slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) {
    return operators::slice<Node>(x, dim, lower, upper);
}

inline Node Node_concat(const std::vector<Node> &xs, unsigned dim) {
    return operators::concat<Node>(xs, dim);
}

inline Node Node_reshape(const Node &x, const Shape &new_shape) {
    return operators::reshape<Node>(x, new_shape);
}

inline Node Node_flatten(const Node &x) {
    return operators::flatten<Node>(x);
}

inline Node Node_transpose(const Node &x) {
    return operators::transpose<Node>(x);
}

inline Node Node_matmul(const Node &a, const Node &b) {
    return operators::matmul<Node>(a, b);
}

inline Node Node_sqrt(const Node &x) {
    return operators::sqrt<Node>(x);
}

inline Node Node_exp(const Node &x) {
    return operators::exp<Node>(x);
}

inline Node Node_log(const Node &x) {
    return operators::log<Node>(x);
}

inline Node Node_pow(const Node &x, float k) {
    return operators::pow<Node>(x, k);
}

inline Node Node_pow(float x, const Node &k) {
    return operators::pow<Node>(x, k);
}

inline Node Node_pow(const Node &x, const Node &k) {
    return operators::pow<Node>(x, k);
}

inline Node Node_tanh(const Node &x) {
    return operators::tanh<Node>(x);
}

inline Node Node_sigmoid(const Node &x) {
    return operators::sigmoid<Node>(x);
}

inline Node Node_softplus(const Node &x) {
    return operators::softplus<Node>(x);
}

inline Node Node_sin(const Node &x) {
    return operators::sin<Node>(x);
}

inline Node Node_cos(const Node &x) {
    return operators::cos<Node>(x);
}

inline Node Node_tan(const Node &x) {
    return operators::tan<Node>(x);
}

inline Node Node_relu(const Node &x) {
    return operators::relu<Node>(x);
}

inline Node Node_lrelu(const Node &x) {
    return operators::lrelu<Node>(x);
}

inline Node Node_prelu(const Node &x, float a) {
    return operators::prelu<Node>(x, a);
}

inline Node Node_elu(const Node &x, float a) {
    return operators::elu<Node>(x, a);
}

inline Node Node_selu(const Node &x, float a, float s) {
    return operators::selu<Node>(x, a, s);
}

inline Node Node_sum(const Node &x, unsigned dim) {
    return operators::sum<Node>(x, dim);
}

inline Node Node_sum_container(const vector<Node> &xs) {
    return operators::sum<vector<Node>>(xs);
}

inline Node Node_mean(const Node &x, unsigned dim) {
    return operators::mean<Node>(x, dim);
}

inline Node Node_mean_container(const vector<Node> &xs) {
    return operators::mean<vector<Node>>(xs);
}

inline Node Node_broadcast(const Node &x, unsigned dim, unsigned size) {
    return operators::broadcast<Node>(x, dim, size);
}

inline Node Node_logsumexp(const Node &x, unsigned dim) {
    return operators::logsumexp<Node>(x, dim);
}

inline Node Node_log_softmax(const Node &x, unsigned dim) {
    return operators::log_softmax<Node>(x, dim);
}

inline Node Node_softmax(const Node &x, unsigned dim) {
    return operators::softmax<Node>(x, dim);
}

inline Node Node_softmax_cross_entropy(const Node &x, const Node &t, unsigned dim) {
    return operators::softmax_cross_entropy<Node>(x, t, dim);
}

inline Node Node_softmax_cross_entropy(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
    return operators::softmax_cross_entropy<Node>(x, ids, dim);
}

inline Node Node_batch_sum(const Node &x) {
    return operators::batch::sum<Node>(x);
}

inline Node Node_batch_mean(const Node &x) {
    return operators::batch::mean<Node>(x);
}

inline Node Node_batch_normalize(const Node &x) {
    return operators::batch::normalize<Node>(x);
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

inline Node Node_zeros(const Shape &shape, Device &dev) {
    return operators::zeros<Node>(shape, dev);
}

inline Node Node_ones(const Shape &shape, Device &dev) {
    return operators::ones<Node>(shape, dev);
}

inline Node Node_identity(unsigned size, Device &dev) {
    return operators::identity<Node>(size, dev);
}

inline Node Node_random_bernoulli(const Shape &shape, float p, Device &dev, Graph &g) {
    return operators::random::bernoulli(shape, p, dev, g);
}

inline Node Node_random_bernoulli(const Shape &shape, float p, Device &dev) {
    return operators::random::bernoulli<Node>(shape, p, dev);
}

inline Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev, Graph &g) {
    return operators::random::uniform(shape, lower, upper, dev, g);
}

inline Node Node_random_uniform(const Shape &shape, float lower, float upper, Device &dev) {
    return operators::random::uniform<Node>(shape, lower, upper, dev);
}

inline Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
    return operators::random::normal(shape, mean, sd, dev, g);
}

inline Node Node_random_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::normal<Node>(shape, mean, sd, dev);
}

inline Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
    return operators::random::log_normal(shape, mean, sd, dev, g);
}

inline Node Node_random_log_normal(const Shape &shape, float mean, float sd, Device &dev) {
    return operators::random::log_normal<Node>(shape, mean, sd, dev);
}

inline Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev, Graph &g) {
    return operators::random::gumbel(shape, mu, beta, dev, g);
}

inline Node Node_random_gumbel(const Shape &shape, float mu, float beta, Device &dev) {
    return operators::random::gumbel<Node>(shape, mu, beta, dev);
}

inline Node Node_dropout(const Node &x, float rate, bool enabled) {
    return operators::dropout<Node>(x, rate, enabled);
}

}

#endif
