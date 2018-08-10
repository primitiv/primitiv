#ifndef PRIMITIV_CORE_SHAPE_OPS_H_
#define PRIMITIV_CORE_SHAPE_OPS_H_

#include <cstdint>
#include <vector>

#include <primitiv/core/shape.h>

namespace primitiv {
namespace shape_ops {

/**
 * Modifies the shape with keeping the number of elements.
 * @param before Source shape.
 * @param after Target shape.
 * @return Calculated shape with dims of `after` and batch size of `before`.
 */
Shape reshape(const Shape &before, const Shape &after);

/**
 * Calculates a flattened shape.
 * @param x A shape.
 * @return Flattened shape.
 */
Shape flatten(const Shape &x);

/**
 * Calculates a shape after the scalar operation.
 * @param x A shape.
 * @param k A scalar shape.
 * @return Calculated shape, that is equivalent to `(x + k).shape()`.
 * @remarks `k` should be a scalar: `Shape({}, batch_size)`.
 */
Shape scalar_op(const Shape &x, const Shape &k);

/**
 * Calculates a shape after the elementwise operation.
 * @param a A shape.
 * @param b Other shape.
 * @return Calculated shape, that is equivalent to `(a + b).shape()`.
 */
Shape elementwise(const Shape &a, const Shape &b);

/**
 * Calculates a shape of the slice.
 * @param x A shape.
 * @param dim Dimension to slice.
 * @param lower Lower bound of the dimension `dim`.
 * @param upper Lower bound of the dimension `dim`.
 * @return Calculated shape.
 */
Shape slice(const Shape &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper);

/**
 * Calculates a concatenated shape.
 * @param xs A list of shapes.
 * @param dim Dimension to be concatenated.
 * @return Calculated shape.
 */
Shape concat(const std::vector<Shape> &xs, std::uint32_t dim);

/**
 * Calculates a concatenated shape.
 * @param xs A list of shapes.
 * @param dim Dimension to be concatenated.
 * @return Calculated shape.
 */
Shape concat(const std::vector<const Shape *> &xs, std::uint32_t dim);

/**
 * Calculates a broadcasted shape.
 * @param x A shape.
 * @param dim Dimension to broadcast.
 * @param size New size of the dimension `dim`.
 * @return Calculated shape.
 */
Shape broadcast(const Shape &x, std::uint32_t dim, std::uint32_t size);

/**
 * Calculates a picked shape.
 * @param x A shape.
 * @param ids Label IDs to be picked from the dimension `dim`.
 * @param dim Dimension to pick.
 * @return Calculated shape.
 */
Shape pick(const Shape &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);

/**
 * Calculates a transposed shape.
 * @param x A shape.
 * @return Calculated shape.
 */
Shape transpose(const Shape &x);

/**
 * Calculates a permuted shape.
 * @param x A shape.
 * @return Calculated shape.
 */
Shape permute_dims(const Shape &x, const std::vector<std::uint32_t> &perm);

/** Calculates a shape of matrix products.
 * @param l Shape of the left hand side.
 * @param r Shape of the right hand side.
 * @return Calculated shape.
 */
Shape matmul(const Shape &l, const Shape &r);

/**
 * Calculates a resulting shape of convolution.
 * @param x Shape of the input tensor.
 * @param w Shape of the filter tensor.
 * @param padding0 Zero-padding width of the first dimension.
 * @param padding1 Zero-padding width of the second dimension.
 * @param stride0 Stride of the first dimension.
 * @param stride1 Stride of the second dimension.
 * @param dilation0 Upscaling factor of the first dimension.
 * @param dilation1 Upscaling factor of the second dimension.
 * @return Calculated shape.
 */
Shape conv2d(
    const Shape &x, const Shape &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1);

/**
 * Calculates a resulting shape of 2-dimensional pooling operations.
 * @param x Input shape.
 * @param window0 Window width of the first dimension.
 * @param window1 Window width of the second dimension.
 * @param padding0 Padding width of the first dimension.
 * @param padding1 Padding width of the second dimension.
 * @param stride0 Stride of the first dimension.
 * @param stride1 Stride of the second dimension.
 * @return Calculated shape.
 */
Shape pool2d(
    const Shape &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1);

/**
 * Calculates a picked shape with the batch addresses.
 * @param x A shape.
 * @param ids Label IDs to be picked from the batch.
 * @return Calculated shape.
 */
Shape batch_pick(const Shape &x, const std::vector<std::uint32_t> &ids);

/**
 * Calculates a shape of the slice along the batch axis.
 * @param x A shape.
 * @param lower Lower bound of the batch.
 * @param upper Lower bound of the batch.
 * @return Calculated shape.
 */
Shape batch_slice(const Shape &x, std::uint32_t lower, std::uint32_t upper);

/**
 * Calculates a shape by concatenating minibatches.
 * @param xs A list of shapes.
 * @return Calculated shape.
 */
Shape batch_concat(const std::vector<Shape> &xs);

/**
 * Calculates a shape by concatenating minibatches.
 * @param xs A list of shapes.
 * @return Calculated shape.
 */
Shape batch_concat(const std::vector<const Shape *> &xs);

}  // namespace shape_ops
}  // namespace primitiv

#endif  // PRIMITIV_CORE_SHAPE_OPS_H_
