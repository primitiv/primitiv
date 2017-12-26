/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_SHAPE_H_
#define PRIMITIV_C_SHAPE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque type of Shape.
 */
typedef struct primitiv_Shape primitiv_Shape;

/**
 * Creates a new Shape object.
 * @param shape Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_new( \
    primitiv_Shape **shape);

/**
 * Creates a new Shape object.
 * @param dims List of the dimension sizes.
 * @param n Length of the dims.
 * @param batch Batch size.
 * @param shape Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_new_with_dims(
    const uint32_t *dims, size_t n, uint32_t batch, primitiv_Shape **shape);

/**
 * Deletes the Shape object.
 * @param shape Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Shape_delete(primitiv_Shape *shape);

/**
 * Returns the size of the i-th dimension.
 * @param shape Pointer of a handler.
 * @param i Dimension number to check.
 * @return Size of the i-th dimension.
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_op_getitem(
    const primitiv_Shape *shape,
    uint32_t i);

/**
 * Returns the dimension array.
 * @param shape Pointer of a handler.
 * @param Pointer of an array to receive the dimensions.
 */
extern PRIMITIV_C_API void primitiv_Shape_dims(
    const primitiv_Shape *shape, uint32_t *array);

/**
 * Returns the depth (length of non-1 dimensions) of the shape.
 * @param shape Pointer of a handler.
 * @return The depth of the shape.
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_depth(
    const primitiv_Shape *shape);

/**
 * Returns the batch size.
 * @param shape Pointer of a handler.
 * @return Batch size.
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_batch(
    const primitiv_Shape *shape);

/**
 * Returns the number of elements in each sample.
 * This value is equal to the product of all dimensions.
 * @param shape Pointer of a handler.
 * @return Number of elements.
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_volume(
    const primitiv_Shape *shape);

/**
 * Returns the number of elements in 1 to specified dim.
 * @param shape Pointer of a handler.
 * @param dim Upper bound of the dimension.
 * @return `dims[0] * dims[1] * ... * dims[dim-1]`
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_lower_volume(
    const primitiv_Shape *shape, uint32_t dim);

/**
 * Returns the number of elements in all samples of the mini-batch.
 * This value is equal to `batch() * volume()`.
 * @param shape Pointer of a handler.
 * @return Number of elements.
 */
extern PRIMITIV_C_API uint32_t primitiv_Shape_size(const primitiv_Shape *shape);

/**
 * Returns a string representation of the shape.
 * The format is: "[n,m,...]xk"
 * @param shape Pointer of a handler.
 * @param string Pointer to receive a char sequence.
 * @param length Pointer to receive a length of the char sequence.
 * @return Encoded string.
 */
extern PRIMITIV_C_API void primitiv_Shape_to_string(
    const primitiv_Shape *shape, char *string, size_t *length);

/**
 * Compares this and other shape.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @return true if this and rhs are same, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_op_eq(
    const primitiv_Shape *shape, const primitiv_Shape *rhs);

/**
 * Compares this and other shape.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @return true if this and rhs are not same, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_op_ne(
    const primitiv_Shape *shape, const primitiv_Shape *rhs);

/**
 * Checks whether the shape has minibatch or not.
 * @param shape Pointer of a handler.
 * @return true if the shape has minibatch, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_has_batch(
    const primitiv_Shape *shape);

/**
 * Checks whether two batch size is compatible (broadcastable) or not.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @return true if both batch size is compatible, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_has_compatible_batch(
    const primitiv_Shape *shape, const primitiv_Shape *rhs);

/**
 * Checks whether the shape is a scalar or not.
 * @param shape Pointer of a handler.
 * @return true if the shape is a scalar, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_is_scalar(
    const primitiv_Shape *shape);

/**
 * Checks whether the shape is a row vector or not.
 * @param shape Pointer of a handler.
 * @return true if the shape is a row vector, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_is_row_vector(
    const primitiv_Shape *shape);

/**
 * Checks whether the shape is a vector or a matrix, or not.
 * @param shape Pointer of a handler.
 * @return true if the shape is a vector or a matrix, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_is_matrix(
    const primitiv_Shape *shape);

/**
 * Checks whether two shapes have completely same dimensions.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @return true if both shape have same dimensions, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_has_same_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs);

/**
 * Checks whether two shapes have same dimensions without an axis.
 * (LOO: leave one out)
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param dim Dimension to be ignored.
 * @return true if both shape have same dimensions regardless the dimension
 *         `dim`, false otherwise.
 */
extern PRIMITIV_C_API bool primitiv_Shape_has_same_loo_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim);

/**
 * Creates a new shape which have one different dimension.
 * @param shape Pointer of a handler.
 * @param dim Dimension to be changed.
 * @param m New size of the dimension `dim`.
 * @param new_shape Pointer for a new shape.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_resize_dim(
    const primitiv_Shape *shape, uint32_t dim, uint32_t m,
    primitiv_Shape **new_shape);

/**
 * Creates a new shape which have specified batch size.
 * @param shape Pointer of a handler.
 * @param batch New batch size.
 * @param new_shape Pointer for a new shape.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_resize_batch(
    const primitiv_Shape *shape, uint32_t batch, primitiv_Shape **new_shape);

/**
 * Directly updates a specified dimension.
 * @param shape Pointer of a handler.
 * @param dim Dimension to be updated.
 * @param m New size of the dimension `dim`.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_update_dim(
    primitiv_Shape *shape, uint32_t dim, uint32_t m);

/**
 * Directly updates the batch size.
 * @param shape Pointer of a handler.
 * @param batch New batch size.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Shape_update_batch(
    primitiv_Shape *shape, uint32_t batch);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_SHAPE_H_
