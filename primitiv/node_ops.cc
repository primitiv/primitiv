// TODO(odashi):
// Write unit tests of node_ops.

#include <config.h>

#include <vector>
#include <primitiv/error.h>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/node_ops.h>
#include <primitiv/shape.h>
#include <primitiv/operators.h>
#include <primitiv/parameter.h>

namespace F = primitiv::functions;

#define REG(g, f, ...) \
  g.add_function(std::unique_ptr<Function>(new F::f), {__VA_ARGS__})

#define REGX(x, f, ...) REG((x).graph(), f, __VA_ARGS__)

namespace {

using primitiv::Node;

// Helper to transform pointers to nodes.
std::vector<Node> ptr_to_obj(const std::vector<const Node *> &xs) {
  std::vector<Node> ret;
  ret.reserve(xs.size());
  for (const Node *x : xs) ret.emplace_back(*x);
  return ret;
}

}  // namespace

namespace primitiv {

template<>
Node operator+(const Node &x) { return REGX(x, Positive(), x); }

template<>
Node operator-(const Node &x) { return REGX(x, Negative(), x); }

template<>
Node operator+(const Node &x, float k) { return REGX(x, AddConst(k), x); }

template<>
Node operator+(float k, const Node &x) { return REGX(x, AddConst(k), x); }

template<>
Node operator+(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, AddScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, AddScalar(), a, b);
  else return REGX(a, Add(), a, b);
}

template<>
Node operator-(const Node &x, float k) { return REGX(x, SubtractConstR(k), x); }

template<>
Node operator-(float k, const Node &x) { return REGX(x, SubtractConstL(k), x); }

template<>
Node operator-(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, SubtractScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, SubtractScalarR(), a, b);
  else return REGX(a, Subtract(), a, b);
}

template<>
Node operator*(const Node &x, float k) { return REGX(x, MultiplyConst(k), x); }

template<>
Node operator*(float k, const Node &x) { return REGX(x, MultiplyConst(k), x); }

template<>
Node operator*(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, MultiplyScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, MultiplyScalar(), a, b);
  else return REGX(a, Multiply(), a, b);
}

template<>
Node operator/(const Node &x, float k) { return REGX(x, DivideConstR(k), x); }

template<>
Node operator/(float k, const Node &x) { return REGX(x, DivideConstL(k), x); }

template<>
Node operator/(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, DivideScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, DivideScalarR(), a, b);
  else return REGX(a, Divide(), a, b);
}

namespace operators {

Node input_node(
    const Shape &shape, const std::vector<float> &data, Device &dev, Graph &g) {
  return REG(g, Input(shape, data, dev));
}

Node input_node(Parameter &param, Graph &g) {
  return REG(g, ParameterInput(param));
}

template<>
Node copy(const Node &x, Device &dev) {
  return REGX(x, Copy(dev), x);
}

template<>
Node pick(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
  return REGX(x, Pick(ids, dim), x);
}

template<>
Node slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) {
  return REGX(x, Slice(dim, lower, upper), x);
}

template<>
Node concat(const std::vector<Node> &xs, unsigned dim) {
  if (xs.empty()) THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_function(
      std::unique_ptr<Function>(new F::Concat(dim)), xs);
}

template<>
Node concat(const std::vector<const Node *> &xs, unsigned dim) {
  return concat(::ptr_to_obj(xs), dim);
}

template<>
Node reshape(const Node &x, const Shape &shape) {
  return REGX(x, Reshape(shape), x);
}

template<>
Node flatten(const Node &x) {
  return REGX(x, Flatten(), x);
}

template<>
Node transpose(const Node &x) {
  return REGX(x, Transpose(), x);
}

template<>
Node matmul(const Node &a, const Node &b) {
  return REGX(a, MatrixMultiply(), a, b);
}

template<>
Node sqrt(const Node &x) {
  return REGX(x, Sqrt(), x);
}

template<>
Node exp(const Node &x) {
  return REGX(x, Exp(), x);
}

template<>
Node tanh(const Node &x) {
  return REGX(x, Tanh(), x);
}

template<>
Node sigmoid(const Node &x) {
  return REGX(x, Sigmoid(), x);
}

template<>
Node softplus(const Node &x) {
  return REGX(x, Softplus(), x);
}

template<>
Node sin(const Node &x) {
  return REGX(x, Sin(), x);
}

template<>
Node cos(const Node &x) {
  return REGX(x, Cos(), x);
}

template<>
Node tan(const Node &x) {
  return REGX(x, Tan(), x);
}

template<>
Node relu(const Node &x) {
  return REGX(x, ReLU(), x);
}

template<>
Node lrelu(const Node &x) {
  return REGX(x, LReLU(), x);
}

template<>
Node prelu(const Node &x, float a) {
  return REGX(x, PReLU(a), x);
}

template<>
Node elu(const Node &x, float a) {
  return REGX(x, ELU(a), x);
}

template<>
Node sum(const Node &x, unsigned dim) {
  return REGX(x, Sum(dim), x);
}

template<>
Node broadcast(const Node &x, unsigned dim, unsigned size) {
  return REGX(x, Broadcast(dim, size), x);
}

template<>
Node logsumexp(const Node &x, unsigned dim) {
  return REGX(x, LogSumExp(dim), x);
}

template<>
Node log_softmax(const Node &x, unsigned dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

template<>
Node softmax(const Node &x, unsigned dim) {
  return exp(log_softmax(x, dim));
}

template<>
Node softmax_cross_entropy(const Node &x, const Node &t, unsigned dim) {
  return REGX(x, SoftmaxCrossEntropy(dim), x, t);
}

template<>
Node softmax_cross_entropy(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
  return REGX(x, SparseSoftmaxCrossEntropy(ids, dim), x);
}

Node dropout(const Node &x, float rate, bool enabled) {
  if (!enabled) return x;
  if (rate == 1.) return 0. * x;
  const float p = 1. - rate;
  return (1. / p) * x * random::bernoulli(x.shape(), p, x.device(), x.graph());
}

namespace batch {

template<>
Node sum(const Node &x) {
  return REGX(x, BatchSum(), x);
}

}  // namespace batch

Node constant_node(const Shape &shape, float k, Device &dev, Graph &g) {
  return REG(g, Constant(shape, k, dev));
}

namespace random {

Node bernoulli(const Shape &shape, float p, Device &dev, Graph &g) {
  return REG(g, RandomBernoulli(shape, p, dev));
}

Node uniform(const Shape &shape, float lower, float upper, Device &dev, Graph &g) {
  return REG(g, RandomUniform(shape, lower, upper, dev));
}

Node normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
  return REG(g, RandomNormal(shape, mean, sd, dev));
}

Node log_normal(const Shape &shape, float mean, float sd, Device &dev, Graph &g) {
  return REG(g, RandomLogNormal(shape, mean, sd, dev));
}

}  // namespace random

}  // namespace operators

}  // namespace primitiv
