#ifndef PRIMITIV_COMPOSITE_FUNCTIONS_H_
#define PRIMITIV_COMPOSITE_FUNCTIONS_H_

#include <primitiv/arithmetic.h>
#include <primitiv/basic_functions.h>

namespace primitiv {
namespace functions {

template<typename Var>
inline type_traits::Identity<Var> pown(const Var &x, std::int32_t k) {
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
  if (idx == 0) return constant<Var>(x.shape(), 1.);
  Var ret;  // temporarily invalid
  for (Var factor = x; ; factor = factor * factor) {
    if (idx & 1) ret = ret.valid() ? ret * factor : factor;
    if (!(idx >>= 1)) break;
  }
  if (k >= 0) return ret;
  else return 1.0 / ret;
}

template<typename Var>
inline type_traits::Identity<Var> selu(
    const Var &x,
    float a = 1.6732632423543772848170429916717,
    float s = 1.0507009873554804934193349852946) {
  return s * elu(x, a);
}

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

namespace batch {

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

/**
 * zeros_tensor(shape, &dev)
 * zeros_node(shape, &dev, &g)
 * zeros<Var>(shape, &dev)
 * zeros<Var>(shape, dev)
 * zeros<Var>(shape)
 */

inline Tensor zeros_tensor(const Shape &shape, Device *dev) {
  return constant_tensor(shape, 0., dev);
}

inline Node zeros_node(const Shape &shape, Device *dev, Graph *g) {
  return constant_node(shape, 0., dev, g);
}

template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape, Device *dev) {
  return constant<Var>(shape, 0., dev);
}

template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape, Device &dev) {
  return constant<Var>(shape, 0., dev);
}

template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape) {
  return constant<Var>(shape, 0.);
}

/**
 * ones_tensor(shape, &dev)
 * ones_node(shape, &dev, &g)
 * ones<Var>(Shape, &dev)
 * ones<Var>(shape, dev)
 * ones<Var>(shape)
 */

inline Tensor ones_tensor(const Shape &shape, Device *dev) {
  return constant_tensor(shape, 1., dev);
}

inline Node ones_node(const Shape &shape, Device *dev, Graph *g) {
  return constant_node(shape, 1., dev, g);
}

template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape, Device *dev) {
  return constant<Var>(shape, 1., dev);
}

template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape, Device &dev) {
  return constant<Var>(shape, 1., dev);
}

template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape) {
  return constant<Var>(shape, 1.);
}

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

#endif  // PRIMITIV_COMPOSITE_FUNCTIONS_H_
