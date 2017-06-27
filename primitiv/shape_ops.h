#ifndef PRIMITIV_SHAPE_OPS_H_
#define PRIMITIV_SHAPE_OPS_H_

#include <vector>
#include <primitiv/shape.h>

namespace primitiv {
namespace shape_ops {

/**
 * Modifies the shape with keeping the number of elements.
 * @param before Source shape.
 * @param after Target shape.
 * @return A shape with dims of `after` and batch size of `before`.
 */
Shape reshape(const Shape &before, const Shape &after);

/**
 * Calculates the flattened shape.
 * @param x A shape.
 * @return Flattened shape.
 */
Shape flatten(const Shape &x);

/**
 * Calculates the shape after the scalar operation.
 * @param x A shape.
 * @param k A scalar shape.
 * @return A shape, that is equivalent to `(x + k).shape()`.
 * @remarks `k` should be a scalar: `Shape({}, batch_size)`.
 */
Shape scalar_op(const Shape &x, const Shape &k);

/**
 * Calculates the shape after the elementwise operation.
 * @param a A shape.
 * @param b Other shape.
 * @return A shape, that is equivalent to `(a + b).shape()`.
 */
Shape elementwise(const Shape &a, const Shape &b);

/**
 * Calculates the shape of the slice.
 * @param x A shape.
 * @param dim Dimension to slice.
 * @param lower Lower bound of the dimension `dim`.
 * @param upper Lower bound of the dimension `dim`.
 * @return A shape.
 */
Shape slice(const Shape &x, unsigned dim, unsigned lower, unsigned upper);

/**
 * Calculates the concatenated shape.
 * @param xs A list of shapes.
 * @param dim Dimension to be concatenated.
 * @return A shape.
 */
Shape concat(const std::vector<const Shape *> &xs, unsigned dim);

/**
 * Calculates the broadcasted shape.
 * @param x A shape.
 * @param dim Dimension to broadcast.
 * @param size New size of the dimension `dim`.
 * @return A shape.
 */
Shape broadcast(const Shape &x, unsigned dim, unsigned size);

/**
 * Calculates the picked shape.
 * @param x A shape.
 * @param dim Dimension to pick.
 * @param ids Label IDs to be picked from the dimension `dim`.
 * @return A shape.
 */
Shape pick(const Shape &x, unsigned dim, const std::vector<unsigned> &ids);

/**
 * Calculates the transposed shape.
 * @param x A shape.
 * @return A shape.
 */
Shape transpose(const Shape &x);

/** Calculates the shape of matrix products.
 * @param l Shape of the left hand side.
 * @param r Shape of the right hand side.
 * @return A shape.
 */
Shape dot(const Shape &l, const Shape &r);

}  // namespace shape_ops
}  // namespace primitiv

#endif  // PRIMITIV_SHAPE_OPS_H_
