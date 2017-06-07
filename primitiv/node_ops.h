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

Node slice(const Node &x, unsigned dim, unsigned lower, unsigned upper);

Node transpose(const Node &x);
Node dot(const Node &a, const Node &b);
Node exp(const Node &x);
Node tanh(const Node &x);
Node sigmoid(const Node &x);
Node relu(const Node &x);

Node batch_sum(const Node &x);

}  // namespace node_ops
}  // namespace primitiv

#endif  // PRIMITIV_NODE_OPS_H_
