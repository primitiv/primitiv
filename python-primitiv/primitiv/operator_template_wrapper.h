#ifndef PYTHON_PRIMITIV_OPERATOR_TEMPLATE_WRAPPER_H_
#define PYTHON_PRIMITIV_OPERATOR_TEMPLATE_WRAPPER_H_

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


inline Node Node_sum(const Node &x, unsigned dim) {
    return operators::sum<Node>(x, dim);
}

inline Node Node_sum_container(const std::vector<Node> &xs) {
    return operators::sum<std::vector<Node>>(xs);
}

inline Node Node_mean(const Node &x, unsigned dim) {
    return operators::mean<Node>(x, dim);
}

inline Node Node_mean_container(const std::vector<Node> &xs) {
    return operators::mean<std::vector<Node>>(xs);
}

inline Tensor Tensor_input_vector(const Shape &shape, const std::vector<float> &data, Device &dev) {
    return operators::input<Tensor>(shape, data, dev);
}

inline Tensor Tensor_parameter(Parameter &param) {
    return operators::parameter<Tensor>(param);
}


inline Tensor Tensor_sum(const Tensor &x, unsigned dim) {
    return operators::sum<Tensor>(x, dim);
}

inline Tensor Tensor_sum_container(const std::vector<Tensor> &xs) {
    return operators::sum<std::vector<Tensor>>(xs);
}

inline Tensor Tensor_mean(const Tensor &x, unsigned dim) {
    return operators::mean<Tensor>(x, dim);
}

inline Tensor Tensor_mean_container(const std::vector<Tensor> &xs) {
    return operators::mean<std::vector<Tensor>>(xs);
}

}  // namespace python_primitiv

#endif  // PYTHON_PRIMITIV_OPERATOR_TEMPLATE_WRAPPER_H_
