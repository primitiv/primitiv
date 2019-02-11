// TODO(odashi):
// Write unit tests of node_ops.

#include <primitiv/config.h>

#include <vector>

#include <primitiv/core/device.h>
#include <primitiv/core/error.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/operator_impl.h>
#include <primitiv/core/parameter.h>

#define REG(g, op, ...) ( \
    (g).add_operator( \
      std::unique_ptr<Operator>(new operators::op), \
      {__VA_ARGS__}))

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
Node positive(const Node &x) { return REGX(x, Positive(), x)[0]; }

template<>
Node negative(const Node &x) { return REGX(x, Negative(), x)[0]; }

template<>
Node add(const Node &x, float k) { return REGX(x, AddConst(k), x)[0]; }

template<>
Node add(float k, const Node &x) { return REGX(x, AddConst(k), x)[0]; }

template<>
Node add(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, AddScalar(), b, a)[0];
  else if (b.shape().is_scalar()) return REGX(a, AddScalar(), a, b)[0];
  else return REGX(a, Add(), a, b)[0];
}

template<>
Node subtract(const Node &x, float k) {
  return REGX(x, SubtractConstR(k), x)[0];
}

template<>
Node subtract(float k, const Node &x) {
  return REGX(x, SubtractConstL(k), x)[0];
}

template<>
Node subtract(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, SubtractScalarL(), b, a)[0];
  else if (b.shape().is_scalar()) return REGX(a, SubtractScalarR(), a, b)[0];
  else return REGX(a, Subtract(), a, b)[0];
}

template<>
Node multiply(const Node &x, float k) {
  return REGX(x, MultiplyConst(k), x)[0];
}

template<>
Node multiply(float k, const Node &x) {
  return REGX(x, MultiplyConst(k), x)[0];
}

template<>
Node multiply(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, MultiplyScalar(), b, a)[0];
  else if (b.shape().is_scalar()) return REGX(a, MultiplyScalar(), a, b)[0];
  else return REGX(a, Multiply(), a, b)[0];
}

template<>
Node divide(const Node &x, float k) { return REGX(x, DivideConstR(k), x)[0]; }

template<>
Node divide(float k, const Node &x) { return REGX(x, DivideConstL(k), x)[0]; }

template<>
Node divide(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, DivideScalarL(), b, a)[0];
  else if (b.shape().is_scalar()) return REGX(a, DivideScalarR(), a, b)[0];
  else return REGX(a, Divide(), a, b)[0];
}

template<>
Node pow(const Node &x, float k) { return REGX(x, PowConstR(k), x)[0]; }

template<>
Node pow(float k, const Node &x) { return REGX(x, PowConstL(k), x)[0]; }

template<>
Node pow(const Node &a, const Node &b) {
  if (a.shape().is_scalar()) return REGX(a, PowScalarL(), b, a)[0];
  else if (b.shape().is_scalar()) return REGX(a, PowScalarR(), a, b)[0];
  else return REGX(a, Pow(), a, b)[0];
}

template<>
Node pown(const Node &x, std::int32_t k) { return REGX(x, PowN(k), x)[0]; }

Node input_node(
    const Shape &shape, const std::vector<float> &data, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      Input(shape, data, Device::get_reference_or_default(dev))
  )[0];
}

Node parameter_node(primitiv::Parameter &param, Graph *g) {
  return REG(Graph::get_reference_or_default(g), Parameter(param))[0];
}

template<>
Node copy(const Node &x, Device *dev) {
  return REGX(x, Copy(Device::get_reference_or_default(dev)), x)[0];
}

template<>
Node pick(
    const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, Pick(ids, dim), x)[0];
}

template<>
Node slice(
    const Node &x, std::uint32_t dim,
    std::uint32_t lower, std::uint32_t upper) {
  return REGX(x, Slice(dim, lower, upper), x)[0];
}

template<>
std::vector<Node> split(const Node &x, std::uint32_t dim, std::uint32_t n) {
  return REGX(x, Split(dim, n), x);
}

template<>
Node concat<Node>(const std::vector<Node> &xs, std::uint32_t dim) {
  if (xs.empty()) PRIMITIV_THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_operator(
      std::unique_ptr<Operator>(new operators::Concat(dim)), xs
  )[0];
}

template<>
Node concat<Node>(const std::vector<const Node *> &xs, std::uint32_t dim) {
  return concat(::ptr_to_obj(xs), dim);
}

template<>
Node reshape(const Node &x, const Shape &shape) {
  return REGX(x, Reshape(shape), x)[0];
}

template<>
Node flatten(const Node &x) {
  return REGX(x, Flatten(), x)[0];
}

template<>
Node transpose(const Node &x) {
  return REGX(x, Transpose(), x)[0];
}

template<>
Node flip(const Node &x, std::uint32_t dim) {
  return REGX(x, Flip(dim), x)[0];
}

template<>
Node permute_dims(const Node &x, const std::vector<std::uint32_t> &perm) {
  return REGX(x, PermuteDims(perm), x)[0];
}

template<>
Node matmul(const Node &a, const Node &b) {
  return REGX(a, MatrixMultiply(), a, b)[0];
}

template<>
Node abs(const Node &x) {
  return REGX(x, Abs(), x)[0];
}

template<>
Node sqrt(const Node &x) {
  return REGX(x, Sqrt(), x)[0];
}

template<>
Node exp(const Node &x) {
  return REGX(x, Exp(), x)[0];
}

template<>
Node log(const Node &x) {
  return REGX(x, Log(), x)[0];
}

template<>
Node tanh(const Node &x) {
  return REGX(x, Tanh(), x)[0];
}

template<>
Node sigmoid(const Node &x) {
  return REGX(x, Sigmoid(), x)[0];
}

template<>
Node softplus(const Node &x) {
  return REGX(x, Softplus(), x)[0];
}

template<>
Node sin(const Node &x) {
  return REGX(x, Sin(), x)[0];
}

template<>
Node cos(const Node &x) {
  return REGX(x, Cos(), x)[0];
}

template<>
Node tan(const Node &x) {
  return REGX(x, Tan(), x)[0];
}

template<>
Node relu(const Node &x) {
  return REGX(x, ReLU(), x)[0];
}

template<>
Node lrelu(const Node &x) {
  return REGX(x, LReLU(), x)[0];
}

template<>
Node prelu(const Node &x, float a) {
  return REGX(x, PReLU(a), x)[0];
}

template<>
Node elu(const Node &x, float a) {
  return REGX(x, ELU(a), x)[0];
}

template<>
Node max(const Node &x, std::uint32_t dim) {
  return REGX(x, Max(dim), x)[0];
}

template<>
Node min(const Node &x, std::uint32_t dim) {
  return REGX(x, Min(dim), x)[0];
}

template<>
Node sum(const Node &x, std::uint32_t dim) {
  return REGX(x, Sum(dim), x)[0];
}

template<>
Node broadcast(const Node &x, std::uint32_t dim, std::uint32_t size) {
  return REGX(x, Broadcast(dim, size), x)[0];
}

template<>
Node logsumexp(const Node &x, std::uint32_t dim) {
  return REGX(x, LogSumExp(dim), x)[0];
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
  return REGX(x, SoftmaxCrossEntropy(dim), x, t)[0];
}

template<>
Node softmax_cross_entropy(
    const Node &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return REGX(x, SparseSoftmaxCrossEntropy(ids, dim), x)[0];
}

template<>
Node stop_gradient(const Node &x) {
  return REGX(x, StopGradient(), x)[0];
}

template<>
Node conv2d(
    const Node &x, const Node &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1) {
  return REGX(
      x,
      Convolution2D(padding0, padding1, stride0, stride1, dilation0, dilation1),
      x, w
  )[0];
}

template<>
Node max_pool2d(
    const Node &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1) {
  return REGX(
      x,
      MaxPooling2D(window0, window1, padding0, padding1, stride0, stride1),
      x
  )[0];
}

namespace batch {

template<>
Node pick(const Node &x, const std::vector<std::uint32_t> &ids) {
  return REGX(x, BatchPick(ids), x)[0];
}

template<>
Node slice(const Node &x, std::uint32_t lower, std::uint32_t upper) {
  return REGX(x, BatchSlice(lower, upper), x)[0];
}

template<>
std::vector<Node> split(const Node &x, std::uint32_t n) {
  return REGX(x, BatchSplit(n), x);
}

template<>
Node concat<Node>(const std::vector<Node> &xs) {
  if (xs.empty()) PRIMITIV_THROW_ERROR("No nodes to concat.");
  return xs[0].graph().add_operator(
      std::unique_ptr<Operator>(new operators::BatchConcat()), xs
  )[0];
}

template<>
Node concat<Node>(const std::vector<const Node *> &xs) {
  return concat(::ptr_to_obj(xs));
}

template<>
Node sum(const Node &x) {
  return REGX(x, BatchSum(), x)[0];
}

}  // namespace batch

Node constant_node(const Shape &shape, float k, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      Constant(shape, k, Device::get_reference_or_default(dev))
  )[0];
}

Node identity_node(std::uint32_t size, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      Identity(size, Device::get_reference_or_default(dev))
  )[0];
}

namespace random {

Node bernoulli_node(
    const Shape &shape, float p, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomBernoulli(shape, p, Device::get_reference_or_default(dev))
  )[0];
}

Node uniform_node(
    const Shape &shape, float lower, float upper, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomUniform(
        shape, lower, upper, Device::get_reference_or_default(dev))
  )[0];
}

Node normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomNormal(shape, mean, sd, Device::get_reference_or_default(dev))
  )[0];
}

Node log_normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g) {
  return REG(
      Graph::get_reference_or_default(g),
      RandomLogNormal(shape, mean, sd, Device::get_reference_or_default(dev))
  )[0];
}

Node gumbel_node(
    const Shape &shape, float mu, float beta, Device *dev, Graph *g) {
  return mu - beta * log(-log(uniform_node(shape, 0., .9999999, dev, g)));
}

}  // namespace random

}  // namespace functions
}  // namespace primitiv
