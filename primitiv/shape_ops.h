#ifndef PRIMITIV_SHAPE_OPS_H_
#define PRIMITIV_SHAPE_OPS_H_

#include <vector>
#include <primitiv/shape.h>

namespace primitiv {
namespace shape_ops {

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
 * Calculated the broadcasted shape.
 * @param x A shape.
 * @param dim Dimension to broadcast.
 * @param size New size of the dimension `dim`.
 * @return A shape.
 */
Shape broadcast(const Shape &x, unsigned dim, unsigned size);

}  // namespace shape_ops
}  // namespace primitiv

#endif  // PRIMITIV_SHAPE_OPS_H_
