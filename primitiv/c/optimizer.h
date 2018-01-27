#ifndef PRIMITIV_C_OPTIMIZER_H_
#define PRIMITIV_C_OPTIMIZER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/model.h>
#include <primitiv/c/parameter.h>

/**
 * Opaque type of Optimizer.
 */
typedef struct primitivOptimizer primitivOptimizer_t;

/**
 * Deletes the Optimizer object.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_delete(
    primitivOptimizer_t *optimizer);

/**
 * Loads configurations from a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the optimizer parameter file.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_load(
    primitivOptimizer_t *optimizer, const char *path);

/**
 * Saves current configurations to a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the file that will store optimizer parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_save(
    const primitivOptimizer_t *optimizer, const char *path);

/**
 * Retrieves current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch Pointer to receive the current epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_epoch(
    const primitivOptimizer_t *optimizer, uint32_t *epoch);

/**
 * Sets current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch New epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_epoch(
    primitivOptimizer_t *optimizer, uint32_t epoch);

/**
 * Retrieves current learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale Pointer to receive the scaling factor.
 * @return Status code.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitiv_Optimizer_get_learning_rate_scaling(
    const primitivOptimizer_t *optimizer, float *scale);

/**
 * Sets learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale New scaling factor.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitiv_Optimizer_set_learning_rate_scaling(
    primitivOptimizer_t *optimizer, float scale);

/**
 * Retrieves current L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength Pointer to receive the current L2 decay strength.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_weight_decay(
    const primitivOptimizer_t *optimizer, float *strength);

/**
 * Sets L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength New L2 decay strength, or 0 to disable L2 decay.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_weight_decay(
    primitivOptimizer_t *optimizer, float strength);

/**
 * Retrieves current gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold Current gradient clipping threshold.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_gradient_clipping(
    const primitivOptimizer_t *optimizer, float *threshold);

/**
 * Sets gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold New clipping threshold, or 0 to disable gradient clipping.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_gradient_clipping(
    primitivOptimizer_t *optimizer, float threshold);

/**
 * Registers a parameter.
 * @param optimizer Pointer of a handler.
 * @param param Parameter to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_parameter(
    primitivOptimizer_t *optimizer, primitivParameter_t *param);

/**
 * Registers multiple parameters.
 * @param optimizer Pointer of a handler.
 * @param params List of parameters to be optimized.
 * @param n Number of parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_parameters(
  primitivOptimizer_t *optimizer, primitivParameter_t **params, size_t n);

/**
 * Registers a model.
 * @param optimizer Pointer of a handler.
 * @param param Model to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_model(
    primitivOptimizer_t *optimizer, primitivModel_t *model);

/**
 * Registers multiple models.
 * @param optimizer Pointer of a handler.
 * @param params List of models to be optimized.
 * @param n Number of models.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_models(
    primitivOptimizer_t *optimizer, primitivModel_t **models, size_t n);

/**
 * Resets all gradients of registered parameters.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_reset_gradients(
  primitivOptimizer_t *optimizer);

/**
 * Updates parameter values.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_update(
    primitivOptimizer_t *optimizer);

/**
 * Gets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Pointer to receive the value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_int_config(
    primitivOptimizer_t *optimizer, const char *key, uint32_t *value);

/**
 * Sets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Configuration value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_int_config(
    primitivOptimizer_t *optimizer, const char *key, uint32_t value);

/**
 * Gets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Pointer to receive the value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_float_config(
    primitivOptimizer_t *optimizer, const char *key, float *value);

/**
 * Sets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Configuration value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_float_config(
    primitivOptimizer_t *optimizer, const char *key, float value);

#endif  // PRIMITIV_C_OPTIMIZER_H_
