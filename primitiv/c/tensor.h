#ifndef PRIMITIV_C_TENSOR_H_
#define PRIMITIV_C_TENSOR_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/shape.h>

/**
 * Opaque type of Tensor.
 */
typedef struct primitivTensor primitivTensor_t;

/**
 * Creates an invalid Tensor object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateTensor(
    primitivTensor_t **newobj);

/**
 * Creates a clone of an existing Tensor object.
 * @param src Pointer to a source Tensor.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCloneTensor(
    const primitivTensor_t *src, primitivTensor_t **newobj);

/**
 * Deletes the Tensor object.
 * @param tensor Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteTensor(
    primitivTensor_t *tensor);

/**
 * Check whether the object is valid or not.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive a result: the true if the object is valid,
 *               false otherwise.
 * @return Status code.
 * @remarks This returns false when the object is created through the default
 *          constructor or the object had been moved.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivIsValidTensor(
    const primitivTensor_t *tensor, PRIMITIV_C_BOOL *retval);

/**
 * Returns the shape of the Tensor.
 * @param tensor Pointer of a handler.
 * @param newobj Pointer to receive a Shape of the Tensor.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetTensorShape(
    const primitivTensor_t *tensor, primitivShape_t **newobj);

/**
 * Returns the Device object related to the internal memory.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive the Device object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetDeviceFromTensor(
    const primitivTensor_t *tensor, primitivDevice_t **retval);

/**
 * Retrieves one internal value in the tensor.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive an internal float value.
 * @return Status code.
 * @remarks This function can be used only when the tensor is a scalar and
 *          non-minibatched (i.e., shape() == Shape()).
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivEvaluateTensorAsFloat(
    const primitivTensor_t *tensor, float *retval);

/**
 * Retrieves internal values in the tensor as a vector.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive a list of the internal values.
 * @param size Pointer to receive the length of the array.
 * @return Status code.
 * @remarks Each resulting values a re ordered by the column-major order, and
 *          the batch size is assumed as the last dimension of the tensor.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivEvaluateTensorAsArray(
    const primitivTensor_t *tensor, float *retval, size_t *size);

/**
 * Retrieves argmax indices along an axis.
 * @param tensor Pointer of a handler.
 * @param dim A specified axis.
 * @param retval Pointer to receive a list of integers that indicates positions
 *               of the maximum values.
 * @param size Pointer to receive the number of the received indices.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetTensorArgmax(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *retval,
    size_t *size);

/**
 * Retrieves argmin indices along an axis.
 * @param tensor Pointer of a handler.
 * @param dim A specified axis.
 * @param retval Pointer to receive a list of integers that indicates positions
 *               of the minimum values.
 * @param size Pointer to receive the number of the received indices.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetTensorArgmin(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *retval,
    size_t *size);

/**
 * Reset internal values using a constant.
 * @param tensor Pointer of a handler.
 * @return Status code.
 * @param k A value to be used to initialize each element.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivResetTensor(
    primitivTensor_t *tensor, float k);

/**
 * Reset internal values using a vector.
 * @param tensor Pointer of a handler.
 * @param values Array of values to be used to initialize each element.
 * @return Status code.
 * @remarks Length of `values` should be equal to `shape().size()`. Each
 *          element should be ordered by the column-major order, and the batch
 *          size is assumed as the last dimension.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivResetTensorByArray(
    primitivTensor_t *tensor, const float *values);

/**
 * Returns a tensor which have the same values and different shape.
 * @param tensor Pointer of a handler.
 * @param new_shape New shape with batch size 1.
 * @param newobj Pointer to receive a new tensor.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivReshapeTensor(
    const primitivTensor_t *tensor, const primitivShape_t *new_shape,
    primitivTensor_t **newobj);

/**
 * Returns a flattened tensor.
 * @param tensor Pointer of a handler.
 * @param newobj Pointer to receive a new tensor.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivFlattenTensor(
    const primitivTensor_t *tensor, primitivTensor_t **newobj);

/**
 * Directly multiplies a constant.
 * @param tensor Pointer of a handler.
 * @param k A constant to multiply.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivMultiplyTensorByConstantInplace(
    primitivTensor_t *tensor, float k);

/**
 * Directly adds a value.
 * @param tensor Pointer of a handler.
 * @param x A tensor to add.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivAddTensorInplace(
    primitivTensor_t *tensor, const primitivTensor_t *x);

/**
 * Directly subtracts a value.
 * @param tensor Pointer of a handler.
 * @param x A tensor to subtract.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSubtractTensorInplace(
    primitivTensor_t *tensor, const primitivTensor_t *x);

#endif  // PRIMITIV_C_TENSOR_H_
