#include <config.h>

#include <vector>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/node.h>
#include <primitiv/shape.h>
#include <primitiv/parameter.h>

namespace primitiv {

Node operator+(const Node &x) {
  return x.graph()->add_function(new functions::Positive(), {x});
}

Node operator-(const Node &x) {
  return x.graph()->add_function(new functions::Negative(), {x});
}

Node operator+(const Node &x, float k) {
  return x.graph()->add_function(new functions::AddConst(k), {x});
}

Node operator+(float k, const Node &x) {
  return x.graph()->add_function(new functions::AddConst(k), {x});
}

Node operator+(const Node &a, const Node &b) {
  return a.graph()->add_function(new functions::Add(), {a, b});
}

Node operator-(const Node &x, float k) {
  return x.graph()->add_function(new functions::SubtractConstR(k), {x});
}

Node operator-(float k, const Node &x) {
  return x.graph()->add_function(new functions::SubtractConstL(k), {x});
}

Node operator-(const Node &a, const Node &b) {
  return a.graph()->add_function(new functions::Subtract(), {a, b});
}

Node operator*(const Node &x, float k) {
  return x.graph()->add_function(new functions::MultiplyConst(k), {x});
}

Node operator*(float k, const Node &x) {
  return x.graph()->add_function(new functions::MultiplyConst(k), {x});
}

Node operator*(const Node &a, const Node &b) {
  return a.graph()->add_function(new functions::Multiply(), {a, b});
}

Node operator/(const Node &x, float k) {
  return x.graph()->add_function(new functions::DivideConstR(k), {x});
}

Node operator/(float k, const Node &x) {
  return x.graph()->add_function(new functions::DivideConstL(k), {x});
}

Node operator/(const Node &a, const Node &b) {
  return a.graph()->add_function(new functions::Divide(), {a, b});
}

namespace node_ops {

Node input(
    Graph *g, Device *dev, const Shape &shape, const std::vector<float> &data) {
  return g->add_function(new functions::Input(shape, dev, data), {});
}

Node parameter(Graph *g, Parameter *param) {
  return g->add_function(new functions::ParameterInput(param), {});
}

Node slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) {
  return x.graph()->add_function(new functions::Slice(dim, lower, upper), {x});
}

Node transpose(const Node &x) {
  return x.graph()->add_function(new functions::Transpose(), {x});
}

Node dot(const Node &a, const Node &b) {
  return a.graph()->add_function(new functions::Dot(), {a, b});
}

Node exp(const Node &x) {
  return x.graph()->add_function(new functions::Exp(), {x});
}

Node tanh(const Node &x) {
  return x.graph()->add_function(new functions::Tanh(), {x});
}

Node sigmoid(const Node &x) {
  return x.graph()->add_function(new functions::Sigmoid(), {x});
}

Node relu(const Node &x) {
  return x.graph()->add_function(new functions::ReLU(), {x});
}

Node batch_sum(const Node &x) {
  return x.graph()->add_function(new functions::BatchSum(), {x});
}

}  // namespace node_ops
}  // namespace primitiv
