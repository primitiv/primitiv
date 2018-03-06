#ifndef PRIMITIV_C_SHAPE_H_
#define PRIMITIV_C_SHAPE_H_

#include <primitiv/c/define.h>

/**
 * Opaque type of Shape.
 */
typedef struct primitivShape primitivShape_t;

/**
 * Creates a new Shape object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateShape(
    primitivShape_t **newobj);

/**
 * Creates a new Shape object.
 * @param dims List of the dimension sizes.
 * @param n Length of the dims.
 * @param batch Batch size.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateShapeWithDims(
    const uint32_t *dims, size_t n, uint32_t batch, primitivShape_t **newobj);

/**
 * Creates a clone of existing Shape object.
 * @param src Pointer to a source Shape.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCloneShape(
    const primitivShape_t *src, primitivShape_t **newobj);

/**
 * Deletes the Shape object.
 * @param shape Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteShape(
    primitivShape_t *shape);

/**
 * Returns the size of the i-th dimension.
 * @param shape Pointer of a handler.
 * @param i Dimension number to check.
 * @param retval Pointer to receive the size of the i-th dimension.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeDimSize(
    const primitivShape_t *shape, uint32_t i, uint32_t *retval);

/**
 * Returns the dimension array.
 * @param shape Pointer of a handler.
 * @param retval Pointer of an array to receive the dimensions.
 * @param size Pointer to receive the number of the dimensions.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeDims(
    const primitivShape_t *shape, uint32_t *retval, size_t *size);

/**
 * Returns the depth (length of non-1 dimensions) of the shape.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive the depth of the shape.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeDepth(
    const primitivShape_t *shape, uint32_t *retval);

/**
 * Returns the batch size.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive the batch size.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeBatchSize(
    const primitivShape_t *shape, uint32_t *retval);

/**
 * Returns the number of elements in each sample.
 * This value is equal to the product of all dimensions.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive the number of elements.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeVolume(
    const primitivShape_t *shape, uint32_t *retval);

/**
 * Returns the number of elements in 1 to specified dim.
 * @param shape Pointer of a handler.
 * @param dim Upper bound of the dimension.
 * @param retval Pointer to receive the number of elements that is equal to
 *               `dims[0] * dims[1] * ... * dims[dim-1]`
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeLowerVolume(
    const primitivShape_t *shape, uint32_t dim, uint32_t *retval);

/**
 * Returns the number of elements in all samples of the mini-batch.
 * This value is equal to `batch() * volume()`.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive the number of elements.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetShapeSize(
    const primitivShape_t *shape, uint32_t *retval);

/**
 * Returns a string representation of the shape.
 * The format is: "[n,m,...]xk"
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive the encoded string.
 * @param size Pointer to receive a length of the char sequence.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivRepresentShapeAsString(
    const primitivShape_t *shape, char *retval, size_t *size);

/**
 * Compares this and other shape.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param retval Pointer to receive a result: true if this and rhs are same,
 *               false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsShapeEqualTo(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval);

/**
 * Compares this and other shape.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param retval Pointer to receive a result: true if this and rhs are not same,
 *               false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsNotShapeEqualTo(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval);

/**
 * Checks whether the shape has minibatch or not.
 * @param shape Pointer of a handler.
 * @param retval Poniter to receive a result: true if the shape has minibatch,
 *               false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivHasShapeBatch(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval);

/**
 * Checks whether two batch size is compatible (broadcastable) or not.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param retval Pointer to receive a result: true if both batch size is
 *               compatible, false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivHasShapeCompatibleBatch(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval);

/**
 * Checks whether the shape is a scalar or not.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive a result: true if the shape is a scalar,
 *               false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsShapeScalar(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval);

/**
 * Checks whether the shape is a column vector or not.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive a result: true if the shape is a column
 *               vector, false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsShapeColumnVector(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval);

/**
 * Checks whether the shape is a vector or a matrix, or not.
 * @param shape Pointer of a handler.
 * @param retval Pointer to receive a result: true if the shape is a vector or
 *               a matrix, false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsShapeMatrix(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval);

/**
 * Checks whether two shapes have completely same dimensions.
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param retval Pointer to receive a result: true if both shape have same
 *               dimensions, false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivHasShapeSameDims(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval);

/**
 * Checks whether two shapes have same dimensions without an axis.
 * (LOO: leave one out)
 * @param shape Pointer of a handler.
 * @param rhs Shape object to compare.
 * @param dim Dimension to be ignored.
 * @param retval Pointer to receive a result: true if both shape have same
 *               dimensions regardless the dimension `dim`, false otherwise.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivHasShapeSameLooDims(
    const primitivShape_t *shape, const primitivShape_t *rhs, uint32_t dim,
    PRIMITIV_C_BOOL *retval);

/**
 * Creates a new shape which have one different dimension.
 * @param shape Pointer of a handler.
 * @param dim Dimension to be changed.
 * @param m New size of the dimension `dim`.
 * @param new_shape Pointer for a new shape.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivResizeShapeDim(
    const primitivShape_t *shape, uint32_t dim, uint32_t m,
    primitivShape_t **new_shape);

/**
 * Creates a new shape which have specified batch size.
 * @param shape Pointer of a handler.
 * @param batch New batch size.
 * @param new_shape Pointer for a new shape.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivResizeShapeBatch(
    const primitivShape_t *shape, uint32_t batch, primitivShape_t **new_shape);

/**
 * Directly updates a specified dimension.
 * @param shape Pointer of a handler.
 * @param dim Dimension to be updated.
 * @param m New size of the dimension `dim`.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivUpdateShapeDim(
    primitivShape_t *shape, uint32_t dim, uint32_t m);

/**
 * Directly updates the batch size.
 * @param shape Pointer of a handler.
 * @param batch New batch size.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivUpdateShapeBatchSize(
    primitivShape_t *shape, uint32_t batch);

#endif  // PRIMITIV_C_SHAPE_H_
