#ifndef PRIMITIV_C_PARAMETER_H_
#define PRIMITIV_C_PARAMETER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/initializer.h>
#include <primitiv/c/shape.h>

/**
 * Opaque type of Parameter.
 */
typedef struct primitivParameter primitivParameter_t;

/**
 * Creates an invalid Parameter object.
 * @param parameter Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_new(
    primitivParameter_t **parameter);

/**
 * Creates a new Parameter object.
 * @param shape The shape of the parameter. The batch size should be 1.
 * @param value List of initial values. Order of elements should be the
 *              column-major (Fortran) order.
 * @param n Number of values.
 * @param device The device object to manage internal memory.
 * @param parameter Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_new_with_values(
    const primitivShape_t *shape, const float *value, size_t n,
    primitivDevice_t *device, primitivParameter_t **parameter);

/**
 * Creates a new Parameter object.
 * @param shape The shape of the parameter. The batch size should be 1.
 * @param init An Initializer object.
 * @param device The device object to manage internal memory.
 * @param parameter Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_new_with_initializer(
    const primitivShape_t *shape, const primitivInitializer_t *initializer,
    primitivDevice_t *device, primitivParameter_t **parameter);

/**
 * Deletes the Parameter object.
 * @param parameter Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_delete(
    primitivParameter_t *parameter);

/**
 * Initializes the Parameter object.
 * @param parameter Pointer of a handler.
 * @param shape The shape of the parameter. The batch size should be 1.
 * @param value List of initial values. Order of elements should be the
 *              column-major (Fortran) order.
 * @param device The device object to manage internal memory.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_init_with_values(
    primitivParameter_t *parameter, const primitivShape_t *shape,
    const float *value, size_t n, primitivDevice_t *device);

/**
 * Initializes the Parameter object.
 * @param parameter Pointer of a handler.
 * @param shape The shape of the parameter. The batch size should be 1.
 * @param init An Initializer object.
 * @param device The device object to manage internal memory.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_init_with_initializer(
    primitivParameter_t *parameter, const primitivShape_t *shape,
    const primitivInitializer_t *initializer, primitivDevice_t *device);

/**
 * Loads parameters from specified file.
 * @param parameter Pointer of a handler.
 * @param path File path to load parameters.
 * @param with_stats Whether or not to load all additional statistics as well
 *                   as parameter values if the file has them.
 * @param device The device object to manage internal memory.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_load(
    primitivParameter_t *parameter, const char *path, PRIMITIV_C_BOOL with_stats,
    primitivDevice_t *device);

/**
 * Saves current parameters into specified file.
 * @param parameter Pointer of a handler.
 * @param path File path to save parameters.
 * @param with_stats Whether or not to save all additional statistics as well
 *                   as parameter values if the parameter object has them.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_save(
    const primitivParameter_t *parameter, const char *path,
    PRIMITIV_C_BOOL with_stats);

/**
 * Returns whether the parameter is valid or not.
 * @param parameter Pointer of a handler.
 * @param valid Pointer to receive a result: true or false w.r.t. the parameter
 *              is valid or not.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_valid(
    const primitivParameter_t *parameter, PRIMITIV_C_BOOL *valid);

/**
 * Set all gradients to 0.
 * @param parameter Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_reset_gradients(
    primitivParameter_t *parameter);

/**
 * Adds a new optional statistics tensor.
 * @param parameter Pointer of a handler.
 * @param name Name of the statistics.
 * @param shape Shape of the tensor.
 * @return Status code.
 * @remarks All elements in the new statistics tensor is initialized by 0.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_add_stats(
    primitivParameter_t *parameter,
    const char *name,
    const primitivShape_t *shape);

/**
 * Checks whether the statistics with name `name` exists or not.
 * @param parameter Pointer of a handler.
 * @param name Name of the statistics.
 * @param has_stats Pointer to receive a result (true if the entry exists,
 *                  false otherwise).
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_has_stats(
    primitivParameter_t *parameter, const char *name,
    PRIMITIV_C_BOOL *has_stats);

/**
 * Returns the shape of the parameter.
 * @param parameter Pointer of a handler.
 * @param shape Pointer to receive a Shape object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_shape(
    const primitivParameter_t *parameter, primitivShape_t **shape);

/**
 * Returns the Device object to manage the internal memory.
 * @param parameter Pointer of a handler.
 * @param device Pointer to receive a reference of the Device object.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_device(
    const primitivParameter_t *parameter, primitivDevice_t **device);

/**
 * Returns the values of the parameter.
 * @param parameter Pointer of a handler.
 * @param tensor Pointer to receive a reference of a tensor representing the
 *               parameter tensor.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_value(
    const primitivParameter_t *parameter, const primitivTensor_t **tensor);

/**
 * Returns the current gradient of the parameter.
 * @param parameter Pointer of a handler.
 * @param tensor Pointer to receive a reference of a tensor representing the
 *               gradient of the value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_gradient(
    const primitivParameter_t *parameter, const primitivTensor_t **tensor);

/**
 * Returns the current opotional statistics tensor specified by given name.
 * @param parameter Pointer of a handler.
 * @param name Name of the statistics.
 * @param tensor Pointer to receive a reference of a tensor.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Parameter_stats(
    const primitivParameter_t *parameter, const char *name,
    const primitivTensor_t **tensor);

#endif  // PRIMITIV_C_PARAMETER_H_
