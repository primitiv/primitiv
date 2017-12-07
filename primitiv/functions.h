#ifndef PRIMITIV_FUNCTIONS_H_
#define PRIMITIV_FUNCTIONS_H_

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>
#include <primitiv/type_traits.h>

namespace primitiv {

class Parameter;

template<typename Var>
type_traits::Identity<Var> operator+(const Var &x);

template<typename Var>
type_traits::Identity<Var> operator-(const Var &x);

template<typename Var>
type_traits::Identity<Var> operator+(const Var &x, float k);

template<typename Var>
type_traits::Identity<Var> operator+(float k, const Var &x);

template<typename Var>
type_traits::Identity<Var> operator+(const Var &a, const Var &b);

template<typename Var>
type_traits::Identity<Var> operator-(const Var &x, float k);

template<typename Var>
type_traits::Identity<Var> operator-(float k, const Var &x);

template<typename Var>
type_traits::Identity<Var> operator-(const Var &a, const Var &b);

template<typename Var>
type_traits::Identity<Var> operator*(const Var &x, float k);

template<typename Var>
type_traits::Identity<Var> operator*(float k, const Var &x);

template<typename Var>
type_traits::Identity<Var> operator*(const Var &a, const Var &b);

template<typename Var>
type_traits::Identity<Var> operator/(const Var &x, float k);

template<typename Var>
type_traits::Identity<Var> operator/(float k, const Var &x);

template<typename Var>
type_traits::Identity<Var> operator/(const Var &a, const Var &b);

namespace functions {

Node input(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev,
    Graph &g);

template<typename Var>
type_traits::Identity<Var> input(
    const Shape &shape,
    const std::vector<float> &data,
    Device &dev = Device::get_default());

Node parameter(Parameter &param, Graph &g);

template<typename Var>
type_traits::Identity<Var> parameter(Parameter &param);

template<typename Var>
type_traits::Identity<Var> copy(
    const Var &x, Device &dev = Device::get_default());

template<typename Var>
type_traits::Identity<Var> pick(
    const Var &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> slice(
    const Var &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper);

template<typename Var>
type_traits::Identity<Var> concat(const std::vector<Var> &xs, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> concat(
    const std::vector<const Var *> &xs, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> concat(
    const std::initializer_list<Var> xs, std::uint32_t dim) {
  return concat(std::vector<Var>(xs), dim);
}

template<typename Var>
type_traits::Identity<Var> concat(
    const std::initializer_list<const Var *> xs, std::uint32_t dim) {
  return concat(std::vector<const Var *>(xs), dim);
}

template<typename Container>
inline type_traits::Reduce<Container> concat(
    const Container &xs, std::uint32_t dim) {
  using Var = type_traits::Reduce<Container>;
  return concat(std::vector<Var>(xs.begin(), xs.end()), dim);
}

template<typename Container>
inline type_traits::ReducePtr<Container> concat(
    const Container &xs, std::uint32_t dim) {
  using Var = type_traits::ReducePtr<Container>;
  return concat(std::vector<const Var *>(xs.begin(), xs.end()), dim);
}

template<typename Var>
type_traits::Identity<Var> reshape(const Var &x, const Shape &new_shape);

template<typename Var>
type_traits::Identity<Var> flatten(const Var &x);

template<typename Var>
type_traits::Identity<Var> transpose(const Var &x);

template<typename Var>
type_traits::Identity<Var> matmul(const Var &a, const Var &b);

template<typename Var>
type_traits::Identity<Var> sqrt(const Var &x);

template<typename Var>
type_traits::Identity<Var> exp(const Var &x);

template<typename Var>
type_traits::Identity<Var> log(const Var &x);

template<typename Var>
inline type_traits::Identity<Var> pow(const Var &x, float k) {
  return exp(k * log(x));
}

template<typename Var>
inline type_traits::Identity<Var> pow(float x, const Var &k) {
  return exp(k * std::log(x));
}

template<typename Var>
inline type_traits::Identity<Var> pow(const Var &x, const Var &k) {
  return exp(k * log(x));
}

template<typename Var>
type_traits::Identity<Var> tanh(const Var &x);

template<typename Var>
type_traits::Identity<Var> sigmoid(const Var &x);

template<typename Var>
type_traits::Identity<Var> softplus(const Var &x);

template<typename Var>
type_traits::Identity<Var> sin(const Var &x);

template<typename Var>
type_traits::Identity<Var> cos(const Var &x);

template<typename Var>
type_traits::Identity<Var> tan(const Var &x);

template<typename Var>
type_traits::Identity<Var> relu(const Var &x);

template<typename Var>
type_traits::Identity<Var> lrelu(const Var &x);

template<typename Var>
type_traits::Identity<Var> prelu(const Var &x, float a);

template<typename Var>
type_traits::Identity<Var> elu(const Var &x, float a);

template<typename Var>
inline type_traits::Identity<Var> selu(
    const Var &x,
    float a = 1.6732632423543772848170429916717,
    float s = 1.0507009873554804934193349852946) {
  return s * elu(x, a);
}

template<typename Var>
type_traits::Identity<Var> sum(const Var &x, std::uint32_t dim);

template<typename Container>
inline type_traits::Reduce<Container> sum(const Container &xs) {
  using Var = type_traits::Reduce<Container>;
  if (xs.empty()) THROW_ERROR("No nodes to sum.");
  auto it = xs.begin();
  Var ret = *it++;
  while (it != xs.end()) ret = ret + *it++;
  return ret;
}

template<typename Container>
inline type_traits::ReducePtr<Container> sum(const Container &xs) {
  using Var = type_traits::ReducePtr<Container>;
  if (xs.empty()) THROW_ERROR("No nodes to sum.");
  auto it = xs.begin();
  Var ret = **it++;
  while (it != xs.end()) ret = ret + **it++;
  return ret;
}

template<typename Var>
inline type_traits::Identity<Var> mean(const Var &x, std::uint32_t dim) {
  return sum(x, dim) / x.shape()[dim];
}

template<typename Container>
inline type_traits::Reduce<Container> mean(const Container &xs) {
  return sum(xs) / xs.size();
}

template<typename Container>
inline type_traits::ReducePtr<Container> mean(const Container &xs) {
  return sum(xs) / xs.size();
}

template<typename Var>
type_traits::Identity<Var> broadcast(const Var &x, std::uint32_t dim, std::uint32_t size);

template<typename Var>
type_traits::Identity<Var> logsumexp(const Var &x, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> log_softmax(const Var &x, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> softmax(const Var &x, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> softmax_cross_entropy(
    const Var &x, const Var &t, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> softmax_cross_entropy(
    const Var &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> stop_gradient(const Var &x);

namespace batch {

template<typename Var>
type_traits::Identity<Var> sum(const Var &x);

template<typename Var>
inline type_traits::Identity<Var> mean(const Var &x) {
  return sum(x) / x.shape().batch();
}

template<typename Var>
inline type_traits::Identity<Var> normalize(const Var &x) {
  if (!x.shape().has_batch()) return x;  // No meaning of normalization.
  const std::uint32_t b = x.shape().batch();
  const float scale = b / (b - 1.);
  const Var m = mean(x);
  const Var v = scale * (mean(x * x) - m * m);
  return (x - m) / sqrt(v + 1e-8);
}

}  // namespace batch

Node constant(const Shape &shape, float k, Device &dev, Graph &g);

inline Node zeros(const Shape &shape, Device &dev, Graph &g) {
  return constant(shape, 0, dev, g);
}

inline Node ones(const Shape &shape, Device &dev, Graph &g) {
  return constant(shape, 1, dev, g);
}

Node identity(std::uint32_t size, Device &dev, Graph &g);

template<typename Var>
type_traits::Identity<Var> constant(
    const Shape &shape, float k,
    Device &dev = Device::get_default());

template<typename Var>
inline type_traits::Identity<Var> zeros(
    const Shape &shape,
    Device &dev = Device::get_default()) {
  return constant<Var>(shape, 0, dev);
}

template<typename Var>
inline type_traits::Identity<Var> ones(
    const Shape &shape,
    Device &dev = Device::get_default()) {
  return constant<Var>(shape, 1, dev);
}

template<typename Var>
type_traits::Identity<Var> identity(
    std::uint32_t size,
    Device &dev = Device::get_default());

template<typename Var>
inline type_traits::Identity<Var> ipow(const Var &x, std::int32_t k) {
  /*
   * NOTE(odashi):
   * std::abs(-0x800..000) generates undefined behavior under 2's complement
   * systems. However, this value should be also evaluated as 0x800..000 by
   * directly casting to std::uint32_t.
   */
  const std::int32_t min_k = std::numeric_limits<std::int32_t>::min();
  std::uint32_t idx = (k == min_k) ? min_k : std::abs(k);
  /*
   * NOTE(odashi):
   * This function is implemented based on an exponentation-by-squaring method
   * and some minor modifications are also included to prevent generating
   * redundant variables.
   */
  if (idx == 0) return ones<Var>(x.shape());
  Var ret;  // temporarily invalid
  for (Var factor = x; ; factor = factor * factor) {
    if (idx & 1) ret = ret.valid() ? ret * factor : factor;
    if (!(idx >>= 1)) break;
  }
  if (k >= 0) return ret;
  else return 1.0 / ret;
}

namespace random {

Node bernoulli(
    const Shape &shape, float p, Device &dev, Graph &g);

Node uniform(
    const Shape &shape, float lower, float upper, Device &dev, Graph &g);

Node normal(
    const Shape &shape, float mean, float sd, Device &dev, Graph &g);

Node log_normal(
    const Shape &shape, float mean, float sd, Device &dev, Graph &g);

Node gumbel(
    const Shape &shape, float mu, float beta, Device &dev, Graph &g);

template<typename Var>
type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p,
    Device &dev = Device::get_default());

template<typename Var>
type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper,
    Device &dev = Device::get_default());

template<typename Var>
type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd,
    Device &dev = Device::get_default());

template<typename Var>
type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd,
    Device &dev = Device::get_default());

template<typename Var>
inline type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta,
    Device &dev = Device::get_default()) {
  return mu - beta * log(-log(uniform<Var>(shape, 0, .9999999, dev)));
}

}  // namespace random

template<typename Var>
inline type_traits::Identity<Var> dropout(
    const Var &x, float rate, bool enabled) {
  if (!enabled) return x;
  if (rate == 1.) return 0. * x;
  const float p = 1. - rate;
  return (1. / p) * x * random::bernoulli<Var>(x.shape(), p, x.device());
}

}  // namespace functions

}  // namespace primitiv

#endif  // PRIMITIV_FUNCTIONS_H_
