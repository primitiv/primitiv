// TODO(odashi):
// Write unit tests of node_ops.

#include <primitiv/config.h>

#include <vector>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/graph.h>
#include <primitiv/shape.h>
#include <primitiv/operator_impl.h>
#include <primitiv/parameter.h>

#define REG(g, op, ...) \
  g.add_operator(std::unique_ptr<Operator>(new operators::op), {__VA_ARGS__})

#define REGX(x, op, ...) REG((x).graph(), op, __VA_ARGS__)

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
namespace functions {

template<>
Node positive(const Node &x) { return REGX(x, Positive(), x); }

template<>
Node negative(const Node &x) { return REGX(x, Negative(), x); }

template<>
Node add(const Node &x, float k) { return REGX(x, AddConst(k), x); }

template<>
Node add(float k, const Node &x) { return REGX(x, AddConst(k), x); }

template<>
Node add(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, AddScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, AddScalar(), a, b);
  else return REGX(a, Add(), a, b);
}

template<>
Node subtract(const Node &x, float k) { return REGX(x, SubtractConstR(k), x); }

template<>
Node subtract(float k, const Node &x) { return REGX(x, SubtractConstL(k), x); }

template<>
Node subtract(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, SubtractScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, SubtractScalarR(), a, b);
  else return REGX(a, Subtract(), a, b);
}

template<>
Node multiply(const Node &x, float k) { return REGX(x, MultiplyConst(k), x); }

template<>
Node multiply(float k, const Node &x) { return REGX(x, MultiplyConst(k), x); }

template<>
Node multiply(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, MultiplyScalar(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, MultiplyScalar(), a, b);
  else return REGX(a, Multiply(), a, b);
}

template<>
Node divide(const Node &x, float k) { return REGX(x, DivideConstR(k), x); }

template<>
Node divide(float k, const Node &x) { return REGX(x, DivideConstL(k), x); }

template<>
Node divide(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, DivideScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, DivideScalarR(), a, b);
  else return REGX(a, Divide(), a, b);
}

template<>
Node pow(const Node &x, float k) { return REGX(x, PowConstR(k), x); }

template<>
Node pow(float k, const Node &x) { return REGX(x, PowConstL(k), x); }

template<>
Node pow(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, PowScalarL(), b, a);
  else if (b.shape().is_scalar()) return REGX(a, PowScalarR(), a, b);
  else return REGX(a, Pow(), a, b);
}

Node input_node(
    const Shape &shape, const std::vector<float> &data, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      Input(shape, data, Device::get_reference_or_default(dev)));
}

Node parameter_node(Parameter &param, Graph *g) {
  return REG(Graph::get_reference_or_default(g), ParameterInput(param));
}

template<>
Node copy(const Node &x, Device *dev) {
  return REGX(x, Copy(Device::get_reference_or_default(dev)), x);
}

template<>
Node pick(
    const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, Pick(ids, dim), x);
}

template<>
Node slice(
    const Node &x, std::uint32_t dim,
    std::uint32_t lower, std::uint32_t upper) {
  return REGX(x, Slice(dim, lower, upper), x);
}

template<>
Node concat(const std::vector<Node> &xs, std::uint32_t dim) {
  if (xs.empty()) THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_operator(
      std::unique_ptr<Operator>(new operators::Concat(dim)), xs);
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
Node softmax_cross_entropy(
    const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, SparseSoftmaxCrossEntropy(ids, dim), x);
}

template<>
Node stop_gradient(const Node &x) {
  return REGX(x, StopGradient(), x);
}

namespace batch {

template<>
Node sum(const Node &x) {
  return REGX(x, BatchSum(), x);
}

}  // namespace batch

Node constant_node(const Shape &shape, float k, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      Constant(shape, k, Device::get_reference_or_default(dev)));
}

Node identity_node(std::uint32_t size, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      IdentityMatrix(size, Device::get_reference_or_default(dev)));
}

namespace random {

Node bernoulli_node(
    const Shape &shape, float p, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomBernoulli(shape, p, Device::get_reference_or_default(dev)));
}

Node uniform_node(
    const Shape &shape, float lower, float upper, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomUniform(
        shape, lower, upper, Device::get_reference_or_default(dev)));
}

Node normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomNormal(shape, mean, sd, Device::get_reference_or_default(dev)));
}

Node log_normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomLogNormal(shape, mean, sd, Device::get_reference_or_default(dev)));
}

Node gumbel_node(
    const Shape &shape, float mu, float beta, Device *dev, Graph *g) {
  return mu - beta * log(-log(uniform_node(shape, 0., .9999999, dev, g)));
}

}  // namespace random

}  // namespace functions
}  // namespace primitiv
