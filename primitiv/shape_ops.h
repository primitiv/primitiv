#ifndef PRIMITIV_SHAPE_OPS_H_
#define PRIMITIV_SHAPE_OPS_H_

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

}  // namespace shape_ops
}  // namespace primitiv

#endif  // PRIMITIV_SHAPE_OPS_H_
