// TODO(odashi):
// Write unit tests of node_ops.

#include <config.h>

#include <vector>
#include <primitiv/error.h>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
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

Node input(
    const Shape &shape, const std::vector<float> &data, Device &dev, Graph &g) {
  return REG(g, Input(shape, data, dev));
}

template<>
Node input<Node>(
    const Shape &shape, const std::vector<float> &data, Device &dev) {
  return input(shape, data, dev, Graph::get_default());
}

Node parameter(Parameter &param, Graph &g) {
  return REG(g, ParameterInput(param));
}

template<>
Node parameter<Node>(Parameter &param) {
  return parameter(param, Graph::get_default());
}

template<>
Node copy(const Node &x, Device &dev) {
  return REGX(x, Copy(dev), x);
}

template<>
Node pick(const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, Pick(ids, dim), x);
}

template<>
Node slice(const Node &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper) {
  return REGX(x, Slice(dim, lower, upper), x);
}

template<>
Node concat(const std::vector<Node> &xs, std::uint32_t dim) {
  if (xs.empty()) THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_function(
      std::unique_ptr<Function>(new F::Concat(dim)), xs);
}

template<>
Node concat(const std::vector<const Node *> &xs, std::uint32_t dim) {
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
Node log(const Node &x) {
  return REGX(x, Log(), x);
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
Node sum(const Node &x, std::uint32_t dim) {
  return REGX(x, Sum(dim), x);
}

template<>
Node broadcast(const Node &x, std::uint32_t dim, std::uint32_t size) {
  return REGX(x, Broadcast(dim, size), x);
}

template<>
Node logsumexp(const Node &x, std::uint32_t dim) {
  return REGX(x, LogSumExp(dim), x);
}

template<>
Node log_softmax(const Node &x, std::uint32_t dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

template<>
Node softmax(const Node &x, std::uint32_t dim) {
  return exp(log_softmax(x, dim));
}

template<>
Node softmax_cross_entropy(const Node &x, const Node &t, std::uint32_t dim) {
  return REGX(x, SoftmaxCrossEntropy(dim), x, t);
}

template<>
Node softmax_cross_entropy(const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, SparseSoftmaxCrossEntropy(ids, dim), x);
}

namespace batch {

template<>
Node sum(const Node &x) {
  return REGX(x, BatchSum(), x);
}

}  // namespace batch

Node constant(const Shape &shape, float k, Device &dev, Graph &g) {
  return REG(g, Constant(shape, k, dev));
}

Node identity(std::uint32_t size, Device &dev, Graph &g) {
  return REG(g, IdentityMatrix(size, dev));
}

template<>
Node constant<Node>(const Shape &shape, float k, Device &dev) {
  return constant(shape, k, dev, Graph::get_default());
}

template<>
Node identity<Node>(std::uint32_t size, Device &dev) {
  return identity(size, dev, Graph::get_default());
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

Node gumbel(const Shape &shape, float mu, float beta, Device &dev, Graph &g) {
  return mu - beta * log(-log(uniform(shape, 0, .9999999, dev, g)));
}

template<>
Node bernoulli<Node>(const Shape &shape, float p, Device &dev) {
  return bernoulli(shape, p, dev, Graph::get_default());
}

template<>
Node uniform<Node>(const Shape &shape, float lower, float upper, Device &dev) {
  return uniform(shape, lower, upper, dev, Graph::get_default());
}

template<>
Node normal<Node>(const Shape &shape, float mean, float sd, Device &dev) {
  return normal(shape, mean, sd, dev, Graph::get_default());
}

template<>
Node log_normal<Node>(const Shape &shape, float mean, float sd, Device &dev) {
  return log_normal(shape, mean, sd, dev, Graph::get_default());
}

}  // namespace random

}  // namespace operators

}  // namespace primitiv
