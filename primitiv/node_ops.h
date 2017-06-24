#ifndef PRIMITIV_NODE_OPS_H_
#define PRIMITIV_NODE_OPS_H_

#include <vector>
#include <primitiv/node.h>

namespace primitiv {

class Device;
class Graph;
class Parameter;

Node operator+(const Node &x);
Node operator-(const Node &x);
Node operator+(const Node &x, float k);
Node operator+(float k, const Node &x);
Node operator+(const Node &a, const Node &b);
Node operator-(const Node &x, float k);
Node operator-(float k, const Node &x);
Node operator-(const Node &a, const Node &b);
Node operator*(const Node &x, float k);
Node operator*(float k, const Node &x);
Node operator*(const Node &a, const Node &b);
Node operator/(const Node &x, float k);
Node operator/(float k, const Node &x);
Node operator/(const Node &a, const Node &b);

namespace node_ops {

Node input(
    Graph *g, Device *dev, const Shape &shape, const std::vector<float> &data);
Node parameter(Graph *g, Parameter *param);

Node copy(const Node &x, Device *dev);

Node random_bernoulli(const Shape &shape, float p, Device *dev, Graph *g);
Node random_uniform(
    const Shape &shape, float lower, float upper, Device *dev, Graph *g);
Node random_normal(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g);

Node pick(const Node &x, unsigned dim, const std::vector<unsigned> &ids);

Node slice(const Node &x, unsigned dim, unsigned lower, unsigned upper);

Node transpose(const Node &x);
Node dot(const Node &a, const Node &b);
Node exp(const Node &x);
Node tanh(const Node &x);
Node sigmoid(const Node &x);
Node relu(const Node &x);

Node dropout(const Node &x, float rate, bool enabled);

Node sum(const Node &x, unsigned dim);
Node logsumexp(const Node &x, unsigned dim);
Node log_softmax(const Node &x, unsigned dim);
Node softmax(const Node &x, unsigned dim);
Node broadcast(const Node &x, unsigned dim, unsigned size);

Node batch_sum(const Node &x);

Node softmax_cross_entropy(const Node &x, const Node &t, unsigned dim);
Node softmax_cross_entropy(
    const Node &x, unsigned dim, const std::vector<unsigned> &ids);

}  // namespace node_ops
}  // namespace primitiv

#endif  // PRIMITIV_NODE_OPS_H_
