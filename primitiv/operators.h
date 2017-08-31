#ifndef PRIMITIV_OPERATORS_H_
#define PRIMITIV_OPERATORS_H_

#include <initializer_list>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/graph.h>

namespace primitiv {

class Parameter;

Tensor operator+(const Tensor &x);
Node operator+(const Node &x);

Tensor operator-(const Tensor &x);
Node operator-(const Node &x);

Tensor operator+(const Tensor &x, float k);
Node operator+(const Node &x, float k);

Tensor operator+(float k, const Tensor &x);
Node operator+(float k, const Node &x);

Tensor operator+(const Tensor &a, const Tensor &b);
Node operator+(const Node &a, const Node &b);

Tensor operator-(const Tensor &x, float k);
Node operator-(const Node &x, float k);

Tensor operator-(float k, const Tensor &x);
Node operator-(float k, const Node &x);

Tensor operator-(const Tensor &a, const Tensor &b);
Node operator-(const Node &a, const Node &b);

Tensor operator*(const Tensor &x, float k);
Node operator*(const Node &x, float k);

Tensor operator*(float k, const Tensor &x);
Node operator*(float k, const Node &x);

Tensor operator*(const Tensor &a, const Tensor &b);
Node operator*(const Node &a, const Node &b);

Tensor operator/(const Tensor &x, float k);
Node operator/(const Node &x, float k);

Tensor operator/(float k, const Tensor &x);
Node operator/(float k, const Node &x);

Tensor operator/(const Tensor &a, const Tensor &b);
Node operator/(const Node &a, const Node &b);

namespace operators {

Tensor input_tensor(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev = Device::get_default_device());
Node input_node(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev = Device::get_default_device(),
    Graph &g = Graph::get_default_graph());

template<class NodeT>
NodeT input(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev = Device::get_default_device());

template<>
inline Tensor input(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev) {
  return input_tensor(shape, data, dev);
}

template<>
inline Node input(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev) {
  return input_node(shape, data, dev);
}

Tensor input_tensor(Parameter &param);
Node input_node(Parameter &param, Graph &g = Graph::get_default_graph());
template<class NodeT>
NodeT input(Parameter &param);

template<class NodeT>
NodeT copy(const NodeT &x, Device &dev = Device::get_default_device());

template<class NodeT>
NodeT pick(const NodeT &x, const std::vector<unsigned> &ids, unsigned dim);

template<class NodeT>
NodeT slice(const NodeT &x, unsigned dim, unsigned lower, unsigned upper);

template<class NodeT>
NodeT concat(const std::vector<NodeT> &xs, unsigned dim);

template<class NodeT>
inline NodeT concat(const std::initializer_list<NodeT> &xs, unsigned dim) {
  return concat(std::vector<NodeT>(xs), dim);
}

template<class NodeT>
NodeT concat_ptr(const std::vector<const NodeT *> &xs, unsigned dim);

template<class NodeT>
inline NodeT concat_ptr(
    const std::initializer_list<const NodeT *> &xs, unsigned dim) {
  return concat_ptr(std::vector<const NodeT *>(xs), dim);
}

template<class NodeT>
NodeT reshape(const NodeT &x, const Shape &new_shape);

template<class NodeT>
NodeT flatten(const NodeT &x);

template<class NodeT>
NodeT transpose(const NodeT &x);

template<class NodeT>
NodeT matmul(const NodeT &a, const NodeT &b);

template<class NodeT>
NodeT sqrt(const NodeT &x);

template<class NodeT>
NodeT exp(const NodeT &x);

template<class NodeT>
NodeT tanh(const NodeT &x);

template<class NodeT>
NodeT sigmoid(const NodeT &x);

template<class NodeT>
NodeT softplus(const NodeT &x);

template<class NodeT>
NodeT sin(const NodeT &x);

template<class NodeT>
NodeT cos(const NodeT &x);

template<class NodeT>
NodeT tan(const NodeT &x);

template<class NodeT>
NodeT relu(const NodeT &x);

template<class NodeT>
NodeT lrelu(const NodeT &x);

template<class NodeT>
NodeT prelu(const NodeT &x, float a);

template<class NodeT>
NodeT elu(const NodeT &x, float a);

template<class NodeT>
NodeT selu(
    const NodeT &x,
    float a = 1.6732632423543772848170429916717,
    float s = 1.0507009873554804934193349852946);

template<class NodeT>
NodeT sum(const NodeT &x, unsigned dim);

template<class NodeT>
NodeT broadcast(const NodeT &x, unsigned dim, unsigned size);

template<class NodeT>
NodeT logsumexp(const NodeT &x, unsigned dim);

template<class NodeT>
NodeT log_softmax(const NodeT &x, unsigned dim);

template<class NodeT>
NodeT softmax(const NodeT &x, unsigned dim);

template<class NodeT>
NodeT softmax_cross_entropy(const NodeT &x, const NodeT &t, unsigned dim);

template<class NodeT>
NodeT softmax_cross_entropy(
    const NodeT &x,
    const std::vector<unsigned> &ids,
    unsigned dim);

namespace batch {

template<class NodeT>
NodeT sum(const NodeT &x);

}  // namespace batch

}  // namespace operators

}  // namespace primitiv

#endif  // PRIMITIV_OPERATORS_H_
