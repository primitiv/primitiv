#ifndef PRIMITIV_BASIC_FUNCTIONS_H_
#define PRIMITIV_BASIC_FUNCTIONS_H_

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <primitiv/error.h>
#include <primitiv/graph.h>
#include <primitiv/tensor.h>
#include <primitiv/type_traits.h>

namespace primitiv {

class Device;
class Parameter;

namespace functions {

/**
 * Applies a unary \f$ + \f$ operation.
 * This function does not change any values of the argument, and returns a copy
 * of it.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ +x \f$.
 */
template<typename Var>
type_traits::Identity<Var> positive(const Var &x);

/**
 * Applies a unary \f$ - \f$ operation.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ -x \f$.
 */
template<typename Var>
type_traits::Identity<Var> negative(const Var &x);

/**
 * Applies an elementwise addition between a variable and a constant.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k A constant \f$ k \f$.
 * @return A variable representing \f$ x + k \f$.
 */
template<typename Var>
type_traits::Identity<Var> add(const Var &x, float k);

/**
 * Applies an elementwise addition between a constant and a variable.
 * @param k A constant \f$ k \f$.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ k + x \f$.
 */
template<typename Var>
type_traits::Identity<Var> add(float k, const Var &x);

/**
 * Applies an elementwise addition between two variables.
 * @param a A variable representing an argument \f$ a \f$.
 * @param b A variable representing an argument \f$ b \f$.
 * @return A variable representing \f$ a + b \f$.
 */
template<typename Var>
type_traits::Identity<Var> add(const Var &a, const Var &b);

/**
 * Applies an elementwise subtraction between a variable and a constant.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k A constant \f$ k \f$.
 * @return A variable representing \f$ x - k \f$.
 */
template<typename Var>
type_traits::Identity<Var> subtract(const Var &x, float k);

/**
 * Applies an elementwise subtraction between a constant and a variable.
 * @param k A constant \f$ k \f$.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ k - x \f$.
 */
template<typename Var>
type_traits::Identity<Var> subtract(float k, const Var &x);

/**
 * Applies an elementwise subtraction between two variables.
 * @param a A variable representing an argument \f$ a \f$.
 * @param b A variable representing an argument \f$ b \f$.
 * @return A variable representing \f$ a - b \f$.
 */
template<typename Var>
type_traits::Identity<Var> subtract(const Var &a, const Var &b);

/**
 * Applies an elementwise multiplication between a variable and a constant.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k A constant \f$ k \f$.
 * @return A variable representing \f$ x \times k \f$.
 */
template<typename Var>
type_traits::Identity<Var> multiply(const Var &x, float k);

/**
 * Applies an elementwise multiplication between a constant and a variable.
 * @param k A constant \f$ k \f$.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ k \times x \f$.
 */
template<typename Var>
type_traits::Identity<Var> multiply(float k, const Var &x);

/**
 * Applies an elementwise multiplication between two variables.
 * @param a A variable representing an argument \f$ a \f$.
 * @param b A variable representing an argument \f$ b \f$.
 * @return A variable representing \f$ a \times b \f$.
 */
template<typename Var>
type_traits::Identity<Var> multiply(const Var &a, const Var &b);

/**
 * Applies an elementwise division between a variable and a constant.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k A constant \f$ k \f$.
 * @return A variable representing \f$ x / k \f$.
 */
template<typename Var>
type_traits::Identity<Var> divide(const Var &x, float k);

/**
 * Applies an elementwise division between a constant and a variable.
 * @param k A constant \f$ k \f$.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ k / x \f$.
 */
template<typename Var>
type_traits::Identity<Var> divide(float k, const Var &x);

/**
 * Applies an elementwise division between two variables.
 * @param a A variable representing an argument \f$ a \f$.
 * @param b A variable representing an argument \f$ b \f$.
 * @return A variable representing \f$ a / b \f$.
 */
template<typename Var>
type_traits::Identity<Var> divide(const Var &a, const Var &b);

/**
 * Applies an elementwise exponentation between a variable and a constant.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k A constant \f$ k \f$.
 * @return A variable representing \f$ x^k \f$.
 */
template<typename Var>
type_traits::Identity<Var> pow(const Var &x, float k);

/**
 * Applies an elementwise exponentation between a constant and a variable.
 * @param k A constant \f$ k \f$.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ k^x \f$.
 */
template<typename Var>
type_traits::Identity<Var> pow(float k, const Var &x);

/**
 * Applies an elementwise exponentation between two variables.
 * @param a A variable representing an argument \f$ a \f$.
 * @param b A variable representing an argument \f$ b \f$.
 * @return A variable representing \f$ a^b \f$.
 */
template<typename Var>
type_traits::Identity<Var> pow(const Var &a, const Var &b);

/**
 * Applies an elementwise exponentation between a variable and an integer
 * constant. This function can be applied correctly when `x` has some negative
 * values.
 * @param x A variable representing an argument \f$ x \f$.
 * @param k An integer constant \f$ k \f$.
 * @return A variable representing \f$ x^k \f$.
 */
template<typename Var>
type_traits::Identity<Var> pown(const Var &x, std::int32_t k);

/**
 * Creates a new Tensor from specific shape and data.
 * @param shape Shape of the new Tensor.
 * @param data Inner data of the new Tensor. `data.size()` should be equal to
 *        `shape.size()` and each data is ordered by the column-major order.
 * @param dev Device to manage inner data of the Tensor, or `nullptr` to use the
 *            default device.
 * @return A new Tensor.
 */
Tensor input_tensor(
    const Shape &shape, const std::vector<float> &data, Device *dev);

/**
 * Creates a new Node from specific shape and data.
 * @param shape Shape of the new Node.
 * @param data Inner data of the new Node. `data.size()` should be equal to
 *             `shape.size()` and each data is ordered by the column-major
 *             order.
 * @param dev Device to manage inner data of the Node, or `nullptr` to use the
 *            default device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node input_node(
    const Shape &shape, const std::vector<float> &data, Device *dev, Graph *g);

/**
 * Creates a new variable from specific shape and data.
 * This function uses a default graph when creating a new Node.
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *             `shape.size()` and each data is ordered by the column-major
 *             order.
 * @param dev Device to manage inner data of the variable, or `nullptr` to use
 *            the default device.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> input(
    const Shape &shape, const std::vector<float> &data, Device *dev);

/// @cond

template<>
inline Tensor input<Tensor>(
    const Shape &shape, const std::vector<float> &data, Device *dev) {
  return input_tensor(shape, data, dev);
}

template<>
inline Node input<Node>(
    const Shape &shape, const std::vector<float> &data, Device *dev) {
  return input_node(shape, data, dev, nullptr);
}

/// @endcond

/**
 * Creates a new variable from specific shape and data.
 * This function uses a default graph when creating a new Node.
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *        `shape.size()` and each data is ordered by the column-major order.
 * @param dev Device to manage inner data of the variable.
 * @return A new variable.
 */
template<typename Var>
inline type_traits::Identity<Var> input(
    const Shape &shape, const std::vector<float> &data, Device &dev) {
  return input<Var>(shape, data, &dev);
}

/**
 * Creates a new variable from specific shape and data.
 * This function uses a default device and graph.
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *        `shape.size()` and each data is ordered by the column-major order.
 * @return A new variable.
 */
template<typename Var>
inline type_traits::Identity<Var> input(
    const Shape &shape, const std::vector<float> &data) {
  return input<Var>(shape, data, nullptr);
}

/**
 * Creates a new Tensor from a specific Parameter.
 * @param param Parameter to be associated with the Tensor.
 * @return A new Tensor.
 */
Tensor parameter_tensor(Parameter &param);

/**
 * Creates a new Node from a specific Parameter.
 * @param param Parameter to be associated with the Node.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node parameter_node(Parameter &param, Graph *g);

/**
 * Creates a new variable from a specific Parameter.
 * This function uses a default graph when creating a new Node.
 * @param param Parameter to be associated with the variable.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> parameter(Parameter &param);

/// @cond

template<>
inline Tensor parameter<Tensor>(Parameter &param) {
  return parameter_tensor(param);
}

template<>
inline Node parameter<Node>(Parameter &param) {
  return parameter_node(param, nullptr);
}

/// @endcond

/**
 * Copies a variable onto a specific device.
 * @param x A variable to be copied.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable managed on `dev`.
 */
template<typename Var>
type_traits::Identity<Var> copy(const Var &x, Device *dev);

/**
 * Copies a variable onto a specific device.
 * @param x A variable to be copied.
 * @param dev Device to manage the new variable.
 * @return A new variable managed on `dev`.
 */
template<typename Var>
inline type_traits::Identity<Var> copy(const Var &x, Device &dev) {
  return copy(x, &dev);
}

/**
 * Copies a variable onto the default device.
 * @param x A variable to be copied.
 * @return A new variable managed on the default device.
 */
template<typename Var>
inline type_traits::Identity<Var> copy(const Var &x) {
  return copy(x, nullptr);
}

/**
 * Lookups subplanes according to the specific axis and addresses. This function
 * can be used to an embedding lookup associated with a fixed vocabulary.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := & \left( \begin{array}{ccc}
 *      1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *    \end{array} \right), \\
 *    \mathrm{pick}(x, [0, 0, 1], 0) & = &
 *      \left( \begin{array}{ccc} 1 & 4 & 7 \end{array} \right),
 *      \left( \begin{array}{ccc} 1 & 4 & 7 \end{array} \right),
 *      \left( \begin{array}{ccc} 2 & 5 & 8 \end{array} \right), \\
 *    \mathrm{pick}(x, [1, 2], 1) & = &
 *      \left( \begin{array}{c} 4 \\ 5 \\ 6 \end{array} \right),
 *      \left( \begin{array}{c} 7 \\ 8 \\ 9 \end{array} \right), \\
 *    \mathrm{pick}(x, [0], 2) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * The minibatch broadcasting rule is applied between the Shape of `x` and the
 * number of values in `ids`:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right),
 *      \left( \begin{array}{ccc}
 *        11 & 14 & 17 \\ 12 & 15 & 18 \\ 13 & 16 & 19
 *      \end{array} \right),
 *      \left( \begin{array}{ccc}
 *        21 & 24 & 27 \\ 22 & 25 & 28 \\ 23 & 26 & 29
 *      \end{array} \right), \\
 *    \mathrm{pick}(x, [0], 1) & = &
 *      \left( \begin{array}{c} 4 \\ 5 \\ 6 \end{array} \right),
 *      \left( \begin{array}{c} 14 \\ 15 \\ 16 \end{array} \right),
 *      \left( \begin{array}{c} 24 \\ 25 \\ 26 \end{array} \right), \\
 *    \mathrm{pick}(x, [0, 1, 2], 1) & = &
 *      \left( \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right),
 *      \left( \begin{array}{c} 14 \\ 15 \\ 16 \end{array} \right),
 *      \left( \begin{array}{c} 27 \\ 28 \\ 29 \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing an original data.
 * @param ids List of subplane IDs according to the axis `dim`. Each value must
 *            be lower than `x.shape()[dim]`.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> pick(
    const Var &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);

/**
 * Extracts a specific range \f$ [L, U) \f$ of subplanes along a specific axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right), \\
 *    \mathrm{slice}(x, 0, 0, 1) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7
 *      \end{array} \right), \\
 *    \mathrm{slice}(x, 1, 1, 3) & = &
 *      \left( \begin{array}{ccc}
 *        4 & 7 \\ 5 & 8 \\ 6 & 9
 *      \end{array} \right), \\
 *    \mathrm{slice}(x, 2, 0, 1) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing an original data.
 * @param dim Axis to be processed.
 * @param lower Lower bound \f$ L \f$ of ``dim``.
 * @param upper Upper bound \f$ U \f$ of ``dim``.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> slice(
    const Var &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper);

/// @cond

template<typename Var>
type_traits::Identity<Var> concat(
    const std::vector<Var> &xs, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> concat(
    const std::vector<const Var *> &xs, std::uint32_t dim);

template<typename Var>
inline type_traits::Identity<Var> concat(
    const std::initializer_list<Var> xs, std::uint32_t dim) {
  return concat(std::vector<Var>(xs), dim);
}

template<typename Var>
inline type_traits::Identity<Var> concat(
    const std::initializer_list<const Var *> xs, std::uint32_t dim) {
  return concat(std::vector<const Var *>(xs), dim);
}

/// @endcond

/**
 * Concatenates multiple variables along specific axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x_1 & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right), \\
 *    x_2 & := &
 *      \left( \begin{array}{ccc}
 *        11 & 14 & 17 \\ 12 & 15 & 18 \\ 13 & 16 & 19
 *      \end{array} \right), \\
 *    \mathrm{concat}([x_1, x_2], 0) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \\
 *        11 & 14 & 17 \\ 12 & 15 & 18 \\ 13 & 16 & 19 \\
 *      \end{array} \right), \\
 *    \mathrm{concat}([x_1, x_2], 1) & = &
 *      \left( \begin{array}{cccccc}
 *        1 & 4 & 7 & 11 & 14 & 17 \\
 *        2 & 5 & 8 & 12 & 15 & 18 \\
 *        3 & 6 & 9 & 13 & 16 & 19 \\
 *      \end{array} \right). \\
 *    \mathrm{concat}([x_1, x_2], 2) & = &
 *      \left( \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\
 *        2 & 5 & 8 \\
 *        3 & 6 & 9 \\
 *      \end{array} \right), \left( \begin{array}{ccc}
 *        11 & 14 & 17 \\
 *        12 & 15 & 18 \\
 *        13 & 16 & 19 \\
 *      \end{array} \right) \right).
 *  \end{array}
 * @f]
 * @param xs Iterable container of variables which have both `begin()` and
 *           `end()` functions that return an iterator.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Container>
inline type_traits::Reduce<Container> concat(
    const Container &xs, std::uint32_t dim) {
  using Var = type_traits::Reduce<Container>;
  return concat(std::vector<Var>(xs.begin(), xs.end()), dim);
}

/**
 * Same as above, but `xs` has pointers of variables.
 * @param xs Iterable container of pointers of variables which have both
 *           `begin()` and `end()` functions that return an iterator.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Container>
inline type_traits::ReducePtr<Container> concat(
    const Container &xs, std::uint32_t dim) {
  using Var = type_traits::ReducePtr<Container>;
  return concat(std::vector<const Var *>(xs.begin(), xs.end()), dim);
}

/**
 * Changes the Shape of the variable.
 * @param x A variable with an old Shape.
 * @param new_shape A new Shape to be applied to the new variable. The volume
 *                  and the minibatch size must be same as that of `x.shape()`.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> reshape(const Var &x, const Shape &new_shape);

/**
 * Changes the Shape of the variable to the column vector.
 * @param x A variable with an old Shape.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> flatten(const Var &x);

/**
 * Applies a matrix transposition.
 * @param x A variable representing an argument \f$ X \f$. The shape of `x`
 *          must be either a scalar, a column vector or a matrix.
 * @return A new variable representing \f$ X^\top \f$.
 */
template<typename Var>
type_traits::Identity<Var> transpose(const Var &x);

/**
 * Applies a matrix multiplication between two matrices.
 * @param a A variable representing an argument \f$ A \f$. The shape of `a`
 *          must be either a scalar, a column vector or a matrix.
 * @param b A variable representing an argument \f$ B \f$. The shape of `b`
 *          must be either a scalar, a column vector or a matrix, and
 *          `b.shape()[0]` must be equal to `a.shape()[1]`.
 * @return A new variable representing \f$ AB \f$.
 */
template<typename Var>
type_traits::Identity<Var> matmul(const Var &a, const Var &b);

/**
 * Applies an elementwise square root function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \sqrt{x} \f$.
 */
template<typename Var>
type_traits::Identity<Var> sqrt(const Var &x);

/**
 * Applies an elementwise exponential function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ e^x \f$.
 */
template<typename Var>
type_traits::Identity<Var> exp(const Var &x);

/**
 * Applies an elementwise natural logarithm function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \ln (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> log(const Var &x);

/**
 * Applies an elementwise hyperbolic tangent function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \tanh (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> tanh(const Var &x);

/**
 * Applies an elementwise logistic sigmoid function:
 * @f[ \mathrm{sigmoid}(x) := \frac{1}{1 + e^{-x}}. @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \mathrm{sigmoid}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> sigmoid(const Var &x);

/**
 * Applies an elementwise softplus function:
 * @f[ \mathrm{softplus}(x) := \ln (1 + e^x). @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \mathrm{softplus}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> softplus(const Var &x);

/**
 * Applies an elementwise sin function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \sin (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> sin(const Var &x);

/**
 * Applies an elementwise cos function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \cos (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> cos(const Var &x);

/**
 * Applies an elementwise tangent function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \tan (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> tan(const Var &x);

/**
 * Applies an elementwise rectified linear unit (ReLU) function:
 * @f[ \mathrm{ReLU}(x) := \max (x, 0). @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \mathrm{ReLU}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> relu(const Var &x);

/**
 * Applies an elementwise leaky ReLU function:
 * @f[ \mathrm{LReLU}(x) := \max (x, 0.01x). @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \mathrm{LReLU}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> lrelu(const Var &x);

/**
 * Applies an elementwise parameterized ReLU function:
 * @f[
 *  \mathrm{PReLU}(x) := \left\{ \begin{array}{ll}
 *    x, & \mathrm{if} \ x \geq 0, \\
 *    \alpha x, & \mathrm{otherwise}.
 *  \end{array} \right.
 * @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @param a A scaling factor \f$ \alpha \f$.
 * @return A variable representing \f$ \mathrm{PReLU}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> prelu(const Var &x, float a);

/**
 * Applies an elementwise exponential linear unit (ELU) function:
 * @f[
 *  \mathrm{ELU}(x) := \left\{ \begin{array}{ll}
 *    x, & \mathrm{if} \ x \geq 0, \\
 *    \alpha (e^x- 1), & \mathrm{otherwise}.
 *  \end{array} \right.
 * @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @param a A scaling factor \f$ \alpha \f$.
 * @return A variable representing \f$ \mathrm{ELU}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> elu(const Var &x, float a);

template<typename Var>
type_traits::Identity<Var> sum(const Var &x, std::uint32_t dim);

template<typename Var>
type_traits::Identity<Var> broadcast(
    const Var &x, std::uint32_t dim, std::uint32_t size);

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

template<typename Var>
type_traits::Identity<Var> conv2d(
    const Var &x, const Var &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1);

template<typename Var>
type_traits::Identity<Var> max_pool2d(
    const Var &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1);

namespace batch {

template<typename Var>
type_traits::Identity<Var> sum(const Var &x);

}  // namespace batch

/**
 * constant_tensor(shape, k, &dev)
 * constant_node(shape, k, &dev, &g)
 * constant<Var>(shape, k, &dev)
 * constant<Var>(shape, k, dev)
 * constant<Var>(shape, k)
 */

Tensor constant_tensor(const Shape &shape, float k, Device *dev);

Node constant_node(const Shape &shape, float k, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> constant(const Shape &shape, float k, Device *dev);

/// @cond

template<>
inline Tensor constant<Tensor>(const Shape &shape, float k, Device *dev) {
  return constant_tensor(shape, k, dev);
}

template<>
inline Node constant<Node>(const Shape &shape, float k, Device *dev) {
  return constant_node(shape, k, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> constant(
    const Shape &shape, float k, Device &dev) {
  return constant<Var>(shape, k, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> constant(const Shape &shape, float k) {
  return constant<Var>(shape, k, nullptr);
}

/**
 * identity_tensor(size, &dev)
 * identity_node(size, &dev, &g)
 * identity<Var>(size, &dev)
 * identity<Var>(size, dev)
 * identity<Var>(size)
 */

Tensor identity_tensor(std::uint32_t size, Device *dev);

Node identity_node(std::uint32_t size, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> identity(std::uint32_t size, Device *dev);

/// @cond

template<>
inline Tensor identity<Tensor>(std::uint32_t size, Device *dev) {
  return identity_tensor(size, dev);
}

template<>
inline Node identity<Node>(std::uint32_t size, Device *dev) {
  return identity_node(size, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> identity(std::uint32_t size, Device &dev) {
  return identity<Var>(size, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> identity(std::uint32_t size) {
  return identity<Var>(size, nullptr);
}

namespace random {

/**
 * bernoulli_tensor(shape, p, &dev)
 * bernoulli_node(shape, p, &dev, &g)
 * bernoulli<Var>(shape, p, &dev)
 * bernoulli<Var>(shape, p, dev)
 * bernoulli<Var>(shape, p)
 */

Tensor bernoulli_tensor(
    const Shape &shape, float p, Device *dev);

Node bernoulli_node(
    const Shape &shape, float p, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p, Device *dev);

/// @cond

template<>
inline Tensor bernoulli<Tensor>(
    const Shape &shape, float p, Device *dev) {
  return bernoulli_tensor(shape, p, dev);
}

template<>
inline Node bernoulli<Node>(
    const Shape &shape, float p, Device *dev) {
  return bernoulli_node(shape, p, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p, Device &dev) {
  return bernoulli<Var>(shape, p, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p) {
  return bernoulli<Var>(shape, p, nullptr);
}

/**
 * uniform_tensor(shape, lower, upper, &dev)
 * uniform_node(shape, lower, upper, &dev, &g)
 * uniform<Var>(shape, lower, upper, &dev)
 * uniform<Var>(shape, lower, upper, dev)
 * uniform<Var>(shape, lower, upper)
 */

Tensor uniform_tensor(
    const Shape &shape, float lower, float upper, Device *dev);

Node uniform_node(
    const Shape &shape, float lower, float upper, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper, Device *dev);

/// @cond

template<>
inline Tensor uniform<Tensor>(
    const Shape &shape, float lower, float upper, Device *dev) {
  return uniform_tensor(shape, lower, upper, dev);
}

template<>
inline Node uniform<Node>(
    const Shape &shape, float lower, float upper, Device *dev) {
  return uniform_node(shape, lower, upper, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper, Device &dev) {
  return uniform<Var>(shape, lower, upper, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper) {
  return uniform<Var>(shape, lower, upper, nullptr);
}

/**
 * normal_tensor(shape, mean, sd, &dev)
 * normal_node(shape, mean, sd, &dev, &g)
 * normal<Var>(shape, mean, sd, &dev)
 * normal<Var>(shape, mean, sd, dev)
 * normal<Var>(shape, mean, sd)
 */

Tensor normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev);

Node normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd, Device *dev);

/// @cond

template<>
inline Tensor normal<Tensor>(
    const Shape &shape, float mean, float sd, Device *dev) {
  return normal_tensor(shape, mean, sd, dev);
}

template<>
inline Node normal<Node>(
    const Shape &shape, float mean, float sd, Device *dev) {
  return normal_node(shape, mean, sd, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd, Device &dev) {
  return normal<Var>(shape, mean, sd, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd) {
  return normal<Var>(shape, mean, sd, nullptr);
}

/**
 * log_normal_tensor(shape, mean, sd, &dev)
 * log_normal_node(shape, mean, sd, &dev, &g)
 * log_normal<Var>(shape, mean, sd, &dev)
 * log_normal<Var>(shape, mean, sd, dev)
 * log_normal<Var>(shape, mean, sd)
 */

Tensor log_normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev);

Node log_normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd, Device &dev);

/// @cond

template<>
inline Tensor log_normal<Tensor>(
    const Shape &shape, float mean, float sd, Device &dev) {
  return log_normal_tensor(shape, mean, sd, &dev);
}

template<>
inline Node log_normal<Node>(
    const Shape &shape, float mean, float sd, Device &dev) {
  return log_normal_node(shape, mean, sd, &dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd, Device &dev) {
  return log_normal<Var>(shape, mean, sd, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd) {
  return log_normal<Var>(shape, mean, sd, nullptr);
}

/**
 * gumbel_tensor(shape, mu, beta, &dev)
 * gumbel_node(shape, mu, beta, &dev, &g)
 * gumbel<Var>(shape, mu, beta, &dev)
 * gumbel<Var>(shape, mu, beta, dev)
 * gumbel<Var>(shape, mu, beta)
 */

Tensor gumbel_tensor(
    const Shape &shape, float mu, float beta, Device *dev);

Node gumbel_node(
    const Shape &shape, float mu, float beta, Device *dev, Graph *g);

template<typename Var>
type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta, Device *dev);

/// @cond

template<>
inline Tensor gumbel<Tensor>(
    const Shape &shape, float mu, float beta, Device *dev) {
  return gumbel_tensor(shape, mu, beta, dev);
}

template<>
inline Node gumbel<Node>(
    const Shape &shape, float mu, float beta, Device *dev) {
  return gumbel_node(shape, mu, beta, dev, nullptr);
}

/// @endcond

template<typename Var>
inline type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta, Device &dev) {
  return gumbel<Var>(shape, mu, beta, &dev);
}

template<typename Var>
inline type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta) {
  return gumbel<Var>(shape, mu, beta, nullptr);
}

}  // namespace random

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_BASIC_FUNCTIONS_H_
