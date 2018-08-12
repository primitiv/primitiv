#ifndef PRIMITIV_CORE_BASIC_FUNCTIONS_H_
#define PRIMITIV_CORE_BASIC_FUNCTIONS_H_

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <primitiv/core/error.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/tensor.h>
#include <primitiv/core/type_traits.h>

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
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *             `shape.size()` and each data is ordered by the column-major
 *             order.
 * @param dev Device to manage inner data of the variable, or `nullptr` to use
 *            the default device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
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
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *        `shape.size()` and each data is ordered by the column-major order.
 * @param dev Device to manage inner data of the variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> input(
    const Shape &shape, const std::vector<float> &data, Device &dev) {
  return input<Var>(shape, data, &dev);
}

/**
 * Creates a new variable from specific shape and data.
 * @param shape Shape of the new variable.
 * @param data Inner data of the new variable. `data.size()` should be equal to
 *        `shape.size()` and each data is ordered by the column-major order.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
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
 * @param param Parameter to be associated with the variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
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

/**
 * Splits a given variable into specified number of partitions along an axis.
 * @param x A variable representing an original data.
 * @param dim Axis to be processed.
 * @param n The number of resulting partitions.
 *          ``n`` should be able to divide ``x.shape()[dim]`` without residue.
 * @return A list of `n` variables. Each variable is identical with:
 *         ``split(x, dim, n)[i] == slice(x, dim, L(i), U(i))``,
 *         where ``L(i) := i * x.shape()[dim] / n``
 *         and ``U(i) := (i + 1) * x.shape()[dim] / n``.
 * @throw primitiv::Error ``n`` can not divide ``s.shape()[dim]`` without residue.
 */
template<typename Var>
std::vector<type_traits::Identity<Var>> split(
    const Var &x, std::uint32_t dim, std::uint32_t n);

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
 *      \end{array} \right), \\
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
 * @param xs Iterable container of variables. `xs` must have both `begin()` and
 *           `end()` functions that return the begin/end iterators.
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
 * @param xs Iterable container of pointers of variables. `xs` must have both
 *           `begin()` and `end()` functions that return the begin/end
 *           iterators.
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
 * Flips elements along an axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right), \\
 *    \mathrm{flip}(x, 0) & = &
 *      \left( \begin{array}{ccc}
 *        3 & 6 & 9 \\ 2 & 5 & 8 \\ 1 & 4 & 7
 *      \end{array} \right), \\
 *    \mathrm{flip}(x, 1) & = &
 *      \left( \begin{array}{c}
 *        7 & 4 & 1 \\ 8 & 5 & 2 \\ 9 & 6 & 3
 *      \end{array} \right), \\
 *    \mathrm{flip}(x, 2) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing an argument \f$ x \f$. The shape of `x`
 *          must be either a scalar, a column vector or a matrix.
 * @return A new variable representing \f$ \mathrm{flip} (x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> flip(const Var &x, std::uint32_t dim);

/**
 * Permutes dimensions of a tensor.
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left(
 *        \left( \begin{array}{cc}
 *          1 & 4 \\ 2 & 5 \\ 3 & 6
 *        \end{array} \right),
 *        \left( \begin{array}{cc}
 *          11 & 14 \\ 12 & 15 \\ 13 & 16
 *        \end{array} \right),
 *        \left( \begin{array}{cc}
 *          21 & 24 \\ 22 & 25 \\ 23 & 26
 *        \end{array} \right),
 *        \left( \begin{array}{cc}
 *          31 & 34 \\ 32 & 35 \\ 33 & 36
 *        \end{array} \right)
 *      \right), \\
 *    \mathrm{permute\_dims}(x, [1, 2, 0]) & = &
 *      \left(
 *        \left( \begin{array}{cccc}
 *          1 & 11 & 21 & 31 \\ 4 & 14 & 24 & 34
 *        \end{array} \right),
 *        \left( \begin{array}{cccc}
 *          2 & 12 & 22 & 32 \\ 5 & 15 & 25 & 35
 *        \end{array} \right),
 *        \left( \begin{array}{cccc}
 *          3 & 13 & 23 & 33 \\ 6 & 16 & 26 & 36
 *        \end{array} \right)
 *      \right), \\
 *    \mathrm{permute\_dims}(x, [2, 0, 1]) & = &
 *      \left(
 *        \left( \begin{array}{ccc}
 *          1 & 2 & 3 \\ 11 & 12 & 13 \\ 21 & 22 & 23 \\ 31 & 32 & 33
 *        \end{array} \right),
 *        \left( \begin{array}{ccc}
 *          4 & 5 & 6 \\ 14 & 15 & 16 \\ 24 & 35 & 36 \\ 34 & 35 & 36
 *        \end{array} \right)
 *      \right), \\
 *    \mathrm{permute\_dims}(x, [0, 1, 2]) & = & x. \\
 *  \end{array}
 * @f]
 * @param x A variable representing an original data.
 * @param perm A list of dimensions for specifying permutation.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> permute_dims(const Var &x, const std::vector<std::uint32_t> &perm);

/**
 * Permutes dimensions of a tensor.
 * @param x A variable representing an original data.
 * @param perm A list of dimensions for specifying permutation.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> permute_dims(const Var &x, const std::initializer_list<std::uint32_t> perm) {
  return permute_dims(x, std::vector<std::uint32_t>(perm));
}

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
 * Applies an elementwise absolute function.
 * @param x A variable representing an argument \f$ x \f$.
 * @return A variable representing \f$ \vert x \vert \f$.
 */
template<typename Var>
type_traits::Identity<Var> abs(const Var &x);

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
 *    \alpha (e^x - 1), & \mathrm{otherwise}.
 *  \end{array} \right.
 * @f]
 * @param x A variable representing an argument \f$ x \f$.
 * @param a A scaling factor \f$ \alpha \f$.
 * @return A variable representing \f$ \mathrm{ELU}(x) \f$.
 */
template<typename Var>
type_traits::Identity<Var> elu(const Var &x, float a);

/**
 * Retrieves maximum values along an axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 6 & 7 \\ 2 & 5 & 9 \\ 3 & 4 & 8
 *      \end{array} \right), \\
 *    \mathrm{max}(x, 0) & = &
 *      \left( \begin{array}{ccc}
 *        3 & 6 & 9
 *      \end{array} \right), \\
 *    \mathrm{max}(x, 1) & = &
 *      \left( \begin{array}{c}
 *        7 \\ 9 \\ 8
 *      \end{array} \right), \\
 *    \mathrm{max}(x, 2) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 6 & 7 \\ 2 & 5 & 9 \\ 3 & 4 & 8
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing values before reduction.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> max(const Var &x, std::uint32_t dim);

/**
 * Retrieves minimum values along an axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 6 & 7 \\ 2 & 5 & 9 \\ 3 & 4 & 8
 *      \end{array} \right), \\
 *    \mathrm{min}(x, 0) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7
 *      \end{array} \right), \\
 *    \mathrm{min}(x, 1) & = &
 *      \left( \begin{array}{c}
 *        1 \\ 2 \\ 3
 *      \end{array} \right), \\
 *    \mathrm{min}(x, 2) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 6 & 7 \\ 2 & 5 & 9 \\ 3 & 4 & 8
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing values before reduction.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> min(const Var &x, std::uint32_t dim);

/**
 * Applies summation along an axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right), \\
 *    \mathrm{sum}(x, 0) & = &
 *      \left( \begin{array}{ccc}
 *        6 & 15 & 24
 *      \end{array} \right), \\
 *    \mathrm{sum}(x, 1) & = &
 *      \left( \begin{array}{c}
 *        12 \\ 15 \\ 18
 *      \end{array} \right), \\
 *    \mathrm{sum}(x, 2) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing values before reduction.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> sum(const Var &x, std::uint32_t dim);

/**
 * Applies broadcasting along an axis.
 * Following examples show how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x_1 & := &
 *      \left( \begin{array}{ccc}
 *        1 & 2 & 3
 *      \end{array} \right), \\
 *    \mathrm{broadcast}(x_1, 0, 3) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 2 & 3 \\ 1 & 2 & 3 \\ 1 & 2 & 3
 *      \end{array} \right), \\
 *    x_2 & := &
 *      \left( \begin{array}{c}
 *        1 \\ 2 \\ 3
 *      \end{array} \right), \\
 *    \mathrm{broadcast}(x_2, 1, 3) & = &
 *      \left( \begin{array}{ccc}
 *        1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3
 *      \end{array} \right), \\
 *    \mathrm{broadcast}(x_2, 2, 3) & = &
 *      \left(
 *        \left( \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right),
 *        \left( \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right),
 *        \left( \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right)
 *      \right).
 *  \end{array}
 * @f]
 * @param x A variable representing values before reduction.
 * @param dim Axis to be processed.
 * @param size New size of the axis `dim`.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> broadcast(
    const Var &x, std::uint32_t dim, std::uint32_t size);

/**
 * Applies a logsumexp reduction along an axis.
 * This function performs similarly to `primitiv::functions::sum` w.r.t. the
 * axis.
 * @param x A variable representing values before expansion.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> logsumexp(const Var &x, std::uint32_t dim);

/**
 * Applies a softmax operation along an axis, and returns the natural logarithm
 * of resulting values.
 * @param x A variable representing original values.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> log_softmax(const Var &x, std::uint32_t dim);

/**
 * Applies a softmax operation along an axis.
 * @param x A variable representing original values.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> softmax(const Var &x, std::uint32_t dim);

/**
 * Applies a softmax cross entropy function between two variables along an axis.
 * @param x A variable representing logit values.
 * @param t A variable representing ground-truth distribution along the axis
 *          `dim`.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> softmax_cross_entropy(
    const Var &x, const Var &t, std::uint32_t dim);

/**
 * Applies a softmax cross entropy function between logits and one-hot
 * distributions along an axis.
 * @param x A variable representing logit values.
 * @param ids List of one-hot IDs along the axis `dim`. Each value must be lower
 *            than `x.shape()[dim]`.
 * @param dim Axis to be processed.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> softmax_cross_entropy(
    const Var &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);

/**
 * Blocks the gradient propagation beyond this function.
 * This function does not modify any values in the input variable, and force to
 * make all gradients \f$ 0 \f$.
 * @param x A variable representing original values.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> stop_gradient(const Var &x);

/**
 * Applies a 2D convolution between two variables.
 * @param x A variable with Shape \f$ [d_0, d_1, c_1] \f$.
 * @param w A variable with Shape \f$ [u_0, u_1, c_1, c_2] \f$.
 * @param padding0 Width of zero-padding along the first axis.
 * @param padding1 Width of zero-padding along the second axis.
 * @param stride0 Stride along the first axis.
 * @param stride1 Stride along the second axis.
 * @param dilation0 Dilation factor along the first axis.
 * @param dilation1 Dilation factor along the second axis.
 * @return A new variable with Shape \f$ [d'_0, d'_1, c_2] \f$. The first and
 *         second dimension are calculated as following:
 * @f[
 *  d'_i := \frac{d_i + 2 \times \mathrm{padding}_i - (u_i - 1) \times \mathrm{dilation}_i + 1}{\mathrm{stride}_i} + 1.
 * @f]
 */
template<typename Var>
type_traits::Identity<Var> conv2d(
    const Var &x, const Var &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1);

/**
 * Applies a 2D max-pooling operation.
 * @param x A variable with Shape \f$ [d_0, d_1, c] \f$.
 * @param window0 Window size along the first axis.
 * @param window1 Window size along the second axis.
 * @param padding0 Width of \f$ -\infty \f$ padding along the first axis.
 * @param padding1 Width of \f$ -\infty \f$ padding along the second axis.
 * @param stride0 Stride along the first axis.
 * @param stride1 Stride along the second axis.
 * @return A new variable with Shape \f$ [d'_0, d'_1, c] \f$. The first and
 *         second dimension are calculated as following:
 * @f[
 *  d'_i := \frac{d_i + 2 \times \mathrm{padding}_i - \mathrm{window}_i}{\mathrm{stride}_i} + 1.
 * @f]
 */
template<typename Var>
type_traits::Identity<Var> max_pool2d(
    const Var &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1);

namespace batch {

/**
 * Selects items from the batch by IDs.
 * @param x A variable representing an original data.
 * @param ids List of batch IDs. Each value must be lower than
 *            `x.shape().batch()`.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> pick(
    const Var &x, const std::vector<std::uint32_t> &ids);

/**
 * Extracts a specific range \f$ [L, U) \f$ of items along the batch axis.
 * @param x A variable representing an original data.
 * @param lower Lower bound \f$ L \f$ of the batch.
 * @param upper Upper bound \f$ U \f$ of the batch.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> slice(
    const Var &x, std::uint32_t lower, std::uint32_t upper);

/**
 * Splits a given variable into specified number of partitions along the batch.
 * @param x A variable representing an original data.
 * @param n The number of resulting partitions.
 *          ``n`` should be able to divide ``x.shape().batch()`` without
 *          residue.
 * @return A list of `n` variables. Each variable is identical with:
 *         ``batch::split(x, n)[i] == batch::slice(x, L(i), U(i))``,
 *         where ``L(i) := i * x.shape().batch() / n``
 *         and ``U(i) := (i + 1) * x.shape().batch() / n``.
 * @throw primitiv::Error ``n`` can not divide ``s.shape().batch()`` without
 *                        residue.
 */
template<typename Var>
std::vector<type_traits::Identity<Var>> split(const Var &x, std::uint32_t n);

/// @cond

template<typename Var>
type_traits::Identity<Var> concat(const std::vector<Var> &xs);

template<typename Var>
type_traits::Identity<Var> concat(const std::vector<const Var *> &xs);

template<typename Var>
inline type_traits::Identity<Var> concat(const std::initializer_list<Var> xs) {
  return concat(std::vector<Var>(xs));
}

template<typename Var>
inline type_traits::Identity<Var> concat(
        const std::initializer_list<const Var *> xs) {
  return concat(std::vector<const Var *>(xs));
}

/// @endcond

/**
 * Concatenates multiple variables along the batch axis.
 * @param xs Iterable container of variables. `xs` must have both `begin()` and
 *           `end()` functions that return the begin/end iterators.
 * @return A new variable.
 */
template<typename Container>
inline type_traits::Reduce<Container> concat(const Container &xs) {
  using Var = type_traits::Reduce<Container>;
  return concat(std::vector<Var>(xs.begin(), xs.end()));
}

/**
 * Same as above, but `xs` has pointers of variables.
 * @param xs Iterable container of pointers of variables. `xs` must have both
 *           `begin()` and `end()` functions that return the begin/end
 *           iterators.
 * @return A new variable.
 */
template<typename Container>
inline type_traits::ReducePtr<Container> concat(const Container &xs) {
  using Var = type_traits::ReducePtr<Container>;
  return concat(std::vector<const Var *>(xs.begin(), xs.end()));
}

/**
 * Applies summation along the minibatch.
 * Following example shows how this function work:
 * @f[
 *  \begin{array}{lcl}
 *    x & := &
 *      \left( \begin{array}{ccc}
 *        1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9
 *      \end{array} \right),
 *      \left( \begin{array}{ccc}
 *        11 & 14 & 17 \\ 12 & 15 & 18 \\ 13 & 16 & 19
 *      \end{array} \right), \\
 *    \mathrm{batch::sum}(x) & = &
 *      \left( \begin{array}{ccc}
 *        12 & 18 & 24 \\ 14 & 20 & 26 \\ 16 & 22 & 28
 *      \end{array} \right).
 *  \end{array}
 * @f]
 * @param x A variable representing values before reduction.
 * @return A new variable.
 */
template<typename Var>
type_traits::Identity<Var> sum(const Var &x);

}  // namespace batch

/**
 * Creates a new Tensor with all values the constant \f$ k \f$.
 * @param shape Shape of the new Tensor.
 * @param k The constant \f$ k \f$ of values in the Tensor.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor constant_tensor(const Shape &shape, float k, Device *dev);

/**
 * Creates a new Node with all values the constant \f$ k \f$.
 * @param shape Shape of the new Node.
 * @param k The constant \f$ k \f$ of values in the Node.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node constant_node(const Shape &shape, float k, Device *dev, Graph *g);

/**
 * Creates a new variable with all values the constant \f$ k \f$.
 * @param shape Shape of the new variable.
 * @param k The constant \f$ k \f$ of values in the variable.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with all values the constant \f$ k \f$.
 * @param shape Shape of the new variable.
 * @param k The constant \f$ k \f$ of values in the variable.
 * @param dev Device to manage the new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> constant(
    const Shape &shape, float k, Device &dev) {
  return constant<Var>(shape, k, &dev);
}

/**
 * Creates a new variable with all values the constant \f$ k \f$.
 * @param shape Shape of the new variable.
 * @param k The constant \f$ k \f$ of values in the variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> constant(const Shape &shape, float k) {
  return constant<Var>(shape, k, nullptr);
}

/**
 * Creates a new Tensor with an \f$ N \f$-dimensional identity matrix.
 * @param size Size \f$ N \f$ of the matrix in the new Tensor.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor identity_tensor(std::uint32_t size, Device *dev);

/**
 * Creates a new Node with an \f$ N \f$-dimensional identity matrix.
 * @param size Size \f$ N \f$ of the matrix in the new Node.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node identity_node(std::uint32_t size, Device *dev, Graph *g);

/**
 * Creates a new variable with an \f$ N \f$-dimensional identity matrix.
 * @param size Size \f$ N \f$ of the matrix in the new Node.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with an \f$ N \f$-dimensional identity matrix.
 * @param size Size \f$ N \f$ of the matrix in the new Node.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> identity(std::uint32_t size, Device &dev) {
  return identity<Var>(size, &dev);
}

/**
 * Creates a new variable with an \f$ N \f$-dimensional identity matrix.
 * @param size Size \f$ N \f$ of the matrix in the new Node.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> identity(std::uint32_t size) {
  return identity<Var>(size, nullptr);
}

namespace random {

/**
 * Creates a new Tensor with values sampled from the Bernoulli distribution.
 * @param shape Shape of the new Tensor.
 * @param p The parameter \f$ p \f$ of the Bernoulli distribution.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor bernoulli_tensor(
    const Shape &shape, float p, Device *dev);

/**
 * Creates a new Node with values sampled from the Bernoulli distribution.
 * @param shape Shape of the new Node.
 * @param p The parameter \f$ p \f$ of the Bernoulli distribution.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node bernoulli_node(
    const Shape &shape, float p, Device *dev, Graph *g);

/**
 * Creates a new variable with values sampled from the Bernoulli distribution.
 * @param shape Shape of the new variable.
 * @param p The parameter \f$ p \f$ of the Bernoulli distribution.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with values sampled from the Bernoulli distribution.
 * @param shape Shape of the new variable.
 * @param p The parameter \f$ p \f$ of the Bernoulli distribution.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p, Device &dev) {
  return bernoulli<Var>(shape, p, &dev);
}

/**
 * Creates a new variable with values sampled from the Bernoulli distribution.
 * @param shape Shape of the new variable.
 * @param p The parameter \f$ p \f$ of the Bernoulli distribution.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> bernoulli(
    const Shape &shape, float p) {
  return bernoulli<Var>(shape, p, nullptr);
}

/**
 * Creates a new Tensor with values sampled from the uniform distribution.
 * @param shape Shape of the new Tensor.
 * @param lower The lower bound \f$ L \f$ of the uniform distribution.
 * @param upper The upper bound \f$ U \f$ of the uniform distribution.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor uniform_tensor(
    const Shape &shape, float lower, float upper, Device *dev);

/**
 * Creates a new Node with values sampled from the uniform distribution.
 * @param shape Shape of the new Node.
 * @param lower The lower bound \f$ L \f$ of the uniform distribution.
 * @param upper The upper bound \f$ U \f$ of the uniform distribution.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node uniform_node(
    const Shape &shape, float lower, float upper, Device *dev, Graph *g);

/**
 * Creates a new variable with values sampled from the uniform distribution.
 * @param shape Shape of the new variable.
 * @param lower The lower bound \f$ L \f$ of the uniform distribution.
 * @param upper The upper bound \f$ U \f$ of the uniform distribution.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with values sampled from the uniform distribution.
 * @param shape Shape of the new variable.
 * @param lower The lower bound \f$ L \f$ of the uniform distribution.
 * @param upper The upper bound \f$ U \f$ of the uniform distribution.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper, Device &dev) {
  return uniform<Var>(shape, lower, upper, &dev);
}

/**
 * Creates a new variable with values sampled from the uniform distribution.
 * @param shape Shape of the new variable.
 * @param lower The lower bound \f$ L \f$ of the uniform distribution.
 * @param upper The upper bound \f$ U \f$ of the uniform distribution.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> uniform(
    const Shape &shape, float lower, float upper) {
  return uniform<Var>(shape, lower, upper, nullptr);
}

/**
 * Creates a new Tensor with values sampled from the normal distribution.
 * @param shape Shape of the new Tensor.
 * @param mean The mean \f$ \mu \f$ of the normal distribution.
 * @param sd The standard deviation \f$ \sigma \f$ of the normal distribution.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev);

/**
 * Creates a new Node with values sampled from the normal distribution.
 * @param shape Shape of the new Node.
 * @param mean The mean \f$ \mu \f$ of the normal distribution.
 * @param sd The standard deviation \f$ \sigma \f$ of the normal distribution.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g);

/**
 * Creates a new variable with values sampled from the normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The mean \f$ \mu \f$ of the normal distribution.
 * @param sd The standard deviation \f$ \sigma \f$ of the normal distribution.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with values sampled from the normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The mean \f$ \mu \f$ of the normal distribution.
 * @param sd The standard deviation \f$ \sigma \f$ of the normal distribution.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd, Device &dev) {
  return normal<Var>(shape, mean, sd, &dev);
}

/**
 * Creates a new variable with values sampled from the normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The mean \f$ \mu \f$ of the normal distribution.
 * @param sd The standard deviation \f$ \sigma \f$ of the normal distribution.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> normal(
    const Shape &shape, float mean, float sd) {
  return normal<Var>(shape, mean, sd, nullptr);
}

/**
 * Creates a new Tensor with values sampled from the log-normal distribution.
 * @param shape Shape of the new Tensor.
 * @param mean The parameter \f$ \mu \f$ of the log-normal distribution.
 * @param sd The parameter \f$ \sigma \f$ of the log-normal distribution.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor log_normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev);

/**
 * Creates a new Node with values sampled from the log-normal distribution.
 * @param shape Shape of the new Node.
 * @param mean The parameter \f$ \mu \f$ of the log-normal distribution.
 * @param sd The parameter \f$ \sigma \f$ of the log-normal distribution.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node log_normal_node(
    const Shape &shape, float mean, float sd, Device *dev, Graph *g);

/**
 * Creates a new variable with values sampled from the log-normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The parameter \f$ \mu \f$ of the log-normal distribution.
 * @param sd The parameter \f$ \sigma \f$ of the log-normal distribution.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with values sampled from the log-normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The parameter \f$ \mu \f$ of the log-normal distribution.
 * @param sd The parameter \f$ \sigma \f$ of the log-normal distribution.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd, Device &dev) {
  return log_normal<Var>(shape, mean, sd, &dev);
}

/**
 * Creates a new variable with values sampled from the log-normal distribution.
 * @param shape Shape of the new variable.
 * @param mean The parameter \f$ \mu \f$ of the log-normal distribution.
 * @param sd The parameter \f$ \sigma \f$ of the log-normal distribution.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> log_normal(
    const Shape &shape, float mean, float sd) {
  return log_normal<Var>(shape, mean, sd, nullptr);
}

/**
 * Creates a new Tensor with values sampled from the Gumbel distribution.
 * @param shape Shape of the new Tensor.
 * @param mu The location parameter \f$ \mu \f$ of the Gumbel distribution.
 * @param beta The scale parameter \f$ \beta \f$ of the Gumbel distribution.
 * @param dev Device to manage the new Tensor, or `nullptr` to use the default
 *            device.
 * @return A new Tensor.
 */
Tensor gumbel_tensor(
    const Shape &shape, float mu, float beta, Device *dev);

/**
 * Creates a new Node with values sampled from the Gumbel distribution.
 * @param shape Shape of the new Node.
 * @param mu The location parameter \f$ \mu \f$ of the Gumbel distribution.
 * @param beta The scale parameter \f$ \beta \f$ of the Gumbel distribution.
 * @param dev Device to manage the new Node, or `nullptr` to use the default
 *            device.
 * @param g Graph to manage the instance of the Node, or `nullptr` to use the
 *          default graph.
 * @return A new Node.
 */
Node gumbel_node(
    const Shape &shape, float mu, float beta, Device *dev, Graph *g);

/**
 * Creates a new variable with values sampled from the Gumbel distribution.
 * @param shape Shape of the new variable.
 * @param mu The location parameter \f$ \mu \f$ of the Gumbel distribution.
 * @param beta The scale parameter \f$ \beta \f$ of the Gumbel distribution.
 * @param dev Device to manage the new variable, or `nullptr` to use the default
 *            device.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
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

/**
 * Creates a new variable with values sampled from the Gumbel distribution.
 * @param shape Shape of the new variable.
 * @param mu The location parameter \f$ \mu \f$ of the Gumbel distribution.
 * @param beta The scale parameter \f$ \beta \f$ of the Gumbel distribution.
 * @param dev Device to manage the new variable.
 * @return A new variable.
 * @remarks This function uses the default graph when specifying Node as the
 *          template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta, Device &dev) {
  return gumbel<Var>(shape, mu, beta, &dev);
}

/**
 * Creates a new variable with values sampled from the Gumbel distribution.
 * @param shape Shape of the new variable.
 * @param mu The location parameter \f$ \mu \f$ of the Gumbel distribution.
 * @param beta The scale parameter \f$ \beta \f$ of the Gumbel distribution.
 * @return A new variable.
 * @remarks This function always uses the default device, and also uses the
 *          default graph when specifying Node as the template variable.
 */
template<typename Var>
inline type_traits::Identity<Var> gumbel(
    const Shape &shape, float mu, float beta) {
  return gumbel<Var>(shape, mu, beta, nullptr);
}

}  // namespace random

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_CORE_BASIC_FUNCTIONS_H_
