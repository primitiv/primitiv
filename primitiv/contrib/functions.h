#ifndef PRIMITIV_CONTRIB_FUNCTIONS_H_
#define PRIMITIV_CONTRIB_FUNCTIONS_H_

#include <limits>

#include <primitiv/core/arithmetic.h>
#include <primitiv/core/basic_functions.h>

namespace primitiv {
namespace functions {

/**
 * Applies an elementwise scaled ELU function:
 * @f[
 *  \mathrm{SELU}(x) := s \times \left\{ \begin{array}{ll}
 *    x, & \mathrm{if} \ x \geq 0, \\
 *    \alpha (e^x - 1), & \mathrm{otherwise}.
 *  \end{array} \right.
 * @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @param a A scaling factor \f$ \alpha \f$.
 * @param s Another scaling factor \f$ s \f$.
 * @return A variable representing \f$ \mathrm{SELU}(x) \f$.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> selu(
    const Var &x,
    float a = 1.6732632423543772848170429916717,
    float s = 1.0507009873554804934193349852946) {
  return s * elu(x, a);
}

/**
 * Applies summation along variables in the container.
 * @param xs Iterable container of variables. `xs` must have both `begin()` and
 *           `end()` functions that return the begin/end iterators.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Container>
inline type_traits::Reduce<Container> sum(const Container &xs) {
  using Var = type_traits::Reduce<Container>;
  if (xs.empty()) PRIMITIV_THROW_ERROR("No nodes to sum.");
  auto it = xs.begin();
  Var ret = *it++;
  while (it != xs.end()) ret = ret + *it++;
  return ret;
}

/**
 * Same as above, but `xs` has pointers of variables.
 * @param xs Iterable container of pointers of variables. `xs` must have both
 *           `begin()` end `end()` functions that return the begin/end
 *           iterators.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Container>
inline type_traits::ReducePtr<Container> sum(const Container &xs) {
  using Var = type_traits::ReducePtr<Container>;
  if (xs.empty()) PRIMITIV_THROW_ERROR("No nodes to sum.");
  auto it = xs.begin();
  Var ret = **it++;
  while (it != xs.end()) ret = ret + **it++;
  return ret;
}

/**
 * Calculates means along an axis.
 * @param x A variable representing values before reduction.
 * @param dim Axis to be processed.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> mean(const Var &x, std::uint32_t dim) {
  return sum(x, dim) / x.shape()[dim];
}

/**
 * Calculates means along variables in the container.
 * @param xs Iterable container of variables. `xs` must have both `begin()` and
 *           `end()` functions that return the begin/end iterators.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Container>
inline type_traits::Reduce<Container> mean(const Container &xs) {
  return sum(xs) / xs.size();
}

/**
 * Same as above, but `xs` has pointers of variables.
 * @param xs Iterable container of pointers of variables. `xs` must have both
 *           `begin()` end `end()` functions that return the begin/end
 *           iterators.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Container>
inline type_traits::ReducePtr<Container> mean(const Container &xs) {
  return sum(xs) / xs.size();
}

namespace batch {

/**
 * Calculates means along the minibatch.
 * @param x A variable representing values before reduction.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> mean(const Var &x) {
  return sum(x) / x.shape().batch();
}

/**
 * Applies the batch normalization:
 * @f[
 *  \begin{array}{rcl}
 *    m_x & := & \frac{1}{B} \sum_{i=1}^{B} x_i, \\
 *    v_x & := & \frac{B}{B - 1} \left( \frac{1}{B} \sum_{i=0}^{B} x_i^2 - m_x^2 \right), \\
 *    \mathrm{batch::normalize}(x) & := & \frac{x - m_x}{\sqrt{v_x + \epsilon}},
 *  \end{array}
 * @f]
 * where \f$ B \f$ is the minibatch size of \f$ x \f$.
 * @param x A variable representing values before normalization.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
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
 * Creates a new Tensor with all values \f$ 0 \f$.
 * @param shape Shape of the new Tensor.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 * @remarks This function is implemented as a composite of some other functions.
 */
inline Tensor zeros_tensor(const Shape &shape, Device *dev) {
  return constant_tensor(shape, 0., dev);
}

/**
 * Creates a new Node with all values \f$ 0 \f$.
 * @param shape Shape of the new Node.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 * @remarks This function is implemented as a composite of some other functions.
 */
inline Node zeros_node(const Shape &shape, Device *dev, Graph *g) {
  return constant_node(shape, 0., dev, g);
}

/**
 * Creates a new variable with all values \f$ 0 \f$.
 * @param shape Shape of the new variable.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            defice.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape, Device *dev) {
  return constant<Var>(shape, 0., dev);
}

/**
 * Creates a new variable with all values \f$ 0 \f$.
 * @param shape Shape of the new variable.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape, Device &dev) {
  return constant<Var>(shape, 0., dev);
}

/**
 * Creates a new variable with all values \f$ 0 \f$.
 * @param shape Shape of the new variable.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> zeros(const Shape &shape) {
  return constant<Var>(shape, 0.);
}

/**
 * Creates a new Tensor with all values \f$ 1 \f$.
 * @param shape Shape of the new Tensor.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 * @remarks This function is implemented as a composite of some other functions.
 */
inline Tensor ones_tensor(const Shape &shape, Device *dev) {
  return constant_tensor(shape, 1., dev);
}

/**
 * Creates a new Node with all values \f$ 1 \f$.
 * @param shape Shape of the new Node.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 * @remarks This function is implemented as a composite of some other functions.
 */
inline Node ones_node(const Shape &shape, Device *dev, Graph *g) {
  return constant_node(shape, 1., dev, g);
}

/**
 * Creates a new variable with all values \f$ 1 \f$.
 * @param shape Shape of the new variable.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            defice.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape, Device *dev) {
  return constant<Var>(shape, 1., dev);
}

/**
 * Creates a new variable with all values \f$ 1 \f$.
 * @param shape Shape of the new variable.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape, Device &dev) {
  return constant<Var>(shape, 1., dev);
}

/**
 * Creates a new variable with all values \f$ 1 \f$.
 * @param shape Shape of the new variable.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
template<typename Var>
inline type_traits::Identity<Var> ones(const Shape &shape) {
  return constant<Var>(shape, 1.);
}

/**
 * Applies the dropout:
 * @f[
 *  \begin{array}{rcl}
 *    w & \sim & \mathrm{Bernoulli}(w; 1 - r), \\
 *    \mathrm{dropout}(x) & := & \frac{1}{1 - r} \times w \times x.
 *  \end{array}
 * @f]
 * @param x A variable representing original values.
 * @param rate The dropout probability \f$ r \f$.
 *             `0` maintains all values and `1` discards all values.
 * @param enabled If `true`, this function applies the operation.
 *                Otherwise, this function performs nothing.
 * @return A new variable.
 * @remarks This function is implemented as a composite of some other functions.
 */
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

#endif  // PRIMITIV_CONTRIB_FUNCTIONS_H_
