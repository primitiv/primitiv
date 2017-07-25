// TODO(odashi):
// Write unit tests of node_ops.

#include <config.h>

#include <vector>
#include <primitiv/error.h>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/node_ops.h>
#include <primitiv/shape.h>
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

Node operator+(const Node &x) { return REGX(x, Positive(), x); }
Node operator-(const Node &x) { return REGX(x, Negative(), x); }

Node operator+(const Node &x, float k) { return REGX(x, AddConst(k), x); }
Node operator+(float k, const Node &x) { return REGX(x, AddConst(k), x); }

Node operator+(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, AddScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, AddScalar(), a, b);
  else return REGX(a, Add(), a, b);
}

Node operator-(const Node &x, float k) { return REGX(x, SubtractConstR(k), x); }
Node operator-(float k, const Node &x) { return REGX(x, SubtractConstL(k), x); }

Node operator-(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, SubtractScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, SubtractScalarR(), a, b);
  else return REGX(a, Subtract(), a, b);
}

Node operator*(const Node &x, float k) { return REGX(x, MultiplyConst(k), x); }
Node operator*(float k, const Node &x) { return REGX(x, MultiplyConst(k), x); }

Node operator*(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, MultiplyScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, MultiplyScalar(), a, b);
  else return REGX(a, Multiply(), a, b);
}

Node operator/(const Node &x, float k) { return REGX(x, DivideConstR(k), x); }
Node operator/(float k, const Node &x) { return REGX(x, DivideConstL(k), x); }

Node operator/(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, DivideScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, DivideScalarR(), a, b);
  else return REGX(a, Divide(), a, b);
}

namespace node_ops {

Node input(const Shape &shape, const std::vector<float> &data, Device &dev, Graph &g) {
  return REG(g, Input(shape, data, dev));
}

Node input(Parameter &param, Graph &g) {
  return REG(g, ParameterInput(param));
}

Node copy(const Node &x, Device &dev) {
  return REGX(x, Copy(dev), x);
}

Node pick(const Node &x, const std::vector<unsigned> &ids, unsigned dim) {
  return REGX(x, Pick(ids, dim), x);
}

Node slice(const Node &x, unsigned dim, unsigned lower, unsigned upper) {
  return REGX(x, Slice(dim, lower, upper), x);
}

Node concat(const std::vector<Node> &xs, unsigned dim) {
  if (xs.empty()) THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_function(
      std::unique_ptr<Function>(new F::Concat(dim)), xs);
}

Node concat_ptr(const std::vector<const Node *> &xs, unsigned dim) {
  return concat(::ptr_to_obj(xs), dim);
}

Node reshape(const Node &x, const Shape &shape) {
  return REGX(x, Reshape(shape), x);
}

Node flatten(const Node &x) {
  return REGX(x, Flatten(), x);
}

Node transpose(const Node &x) {
  return REGX(x, Transpose(), x);
}

Node matmul(const Node &a, const Node &b) {
  return REGX(a, MatrixMultiply(), a, b);
}

Node sqrt(const Node &x) {
  return REGX(x, Sqrt(), x);
}

Node exp(const Node &x) {
  return REGX(x, Exp(), x);
}

Node tanh(const Node &x) {
  return REGX(x, Tanh(), x);
}

Node sigmoid(const Node &x) {
  return REGX(x, Sigmoid(), x);
}

Node softplus(const Node &x) {
  return REGX(x, Softplus(), x);
}

Node sin(const Node &x) {
  return REGX(x, Sin(), x);
}

Node cos(const Node &x) {
  return REGX(x, Cos(), x);
}

Node tan(const Node &x) {
  return REGX(x, Tan(), x);
}

Node relu(const Node &x) {
  return REGX(x, ReLU(), x);
}

Node lrelu(const Node &x) {
  return REGX(x, LReLU(), x);
}

Node prelu(const Node &x, float a) {
  return REGX(x, PReLU(a), x);
}

Node elu(const Node &x, float a) {
  return REGX(x, ELU(a), x);
}

Node selu(const Node &x, float a, float s) {
  return s * elu(x, a);
}

Node sum(const Node &x, unsigned dim) {
  return REGX(x, Sum(dim), x);
}

Node sum(const std::vector<Node> &xs) {
  if (xs.empty()) THROW_ERROR("No nodes to sum.");
  Node ret = xs[0];
  for (unsigned i = 1; i < xs.size(); ++i) ret = ret + xs[i];
  return ret;
}

Node sum_ptr(const std::vector<const Node *> &xs) {
  return sum(::ptr_to_obj(xs));
}

Node mean(const Node &x, unsigned dim) {
  return (1. / x.shape()[dim]) * sum(x, dim);
}

Node mean(const std::vector<Node> &xs) {
  return sum(xs) / xs.size();
}

Node mean_ptr(const std::vector<const Node *> &xs) {
  return mean(::ptr_to_obj(xs));
}

Node logsumexp(const Node &x, unsigned dim) {
  return REGX(x, LogSumExp(dim), x);
}

Node log_softmax(const Node &x, unsigned dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

Node softmax(const Node &x, unsigned dim) {
  return exp(log_softmax(x, dim));
}

Node broadcast(const Node &x, unsigned dim, unsigned size) {
  return REGX(x, Broadcast(dim, size), x);
}

Node softmax_cross_entropy(const Node &x, const Node &t, unsigned dim) {
  return REGX(x, SoftmaxCrossEntropy(dim), x, t);
}

Node softmax_cross_entropy(const Node &x, unsigned dim, const std::vector<unsigned> &ids) {
  return REGX(x, SparseSoftmaxCrossEntropy(dim, ids), x);
}

Node dropout(const Node &x, float rate, bool enabled) {
  if (!enabled) return x;
  if (rate == 1.) return 0. * x;
  const float p = 1. - rate;
  return (1. / p) * x * random::bernoulli(x.shape(), p, x.device(), x.graph());
}

namespace batch {

Node sum(const Node &x) {
  return REGX(x, BatchSum(), x);
}

Node mean(const Node &x) {
  return (1. / x.shape().batch()) * sum(x);
}

Node normalize(const Node &x) {
  if (!x.shape().has_batch()) return x;  // No meaning of normalization.
  const unsigned b = x.shape().batch();
  const float scale = b / (b - 1.);
  const Node m = mean(x);
  const Node v = scale * (mean(x * x) - m * m);
  return (x - m) / sqrt(v + 1e-8);
}

}  // namespace batch

Node zeros(const Shape &shape, Device &dev, Graph &g) {
  return REG(g, Constant(shape, 0, dev));
}

Node ones(const Shape &shape, Device &dev, Graph &g) {
  return REG(g, Constant(shape, 1, dev));
}

Node constant(const Shape &shape, float k, Device &dev, Graph &g) {
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

}  // namespace node_ops
}  // namespace primitiv
