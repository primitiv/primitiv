#ifndef PRIMITIV_C_OPTIMIZER_H_
#define PRIMITIV_C_OPTIMIZER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/model.h>
#include <primitiv/c/parameter.h>

/**
 * Opaque type of Optimizer.
 */
typedef struct primitiv_Optimizer primitiv_Optimizer;

/**
 * Deletes the Optimizer object.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_delete(
    primitiv_Optimizer *optimizer);

/**
 * Loads configurations from a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the optimizer parameter file.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_load(
    primitiv_Optimizer *optimizer, const char *path);

/**
 * Saves current configurations to a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the file that will store optimizer parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_save(
    const primitiv_Optimizer *optimizer, const char *path);

/**
 * Retrieves current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch Pointer to receive the current epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_epoch(
    const primitiv_Optimizer *optimizer, uint32_t *epoch);

/**
 * Sets current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch New epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_epoch(
    primitiv_Optimizer *optimizer, uint32_t epoch);

/**
 * Retrieves current learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale Pointer to receive the scaling factor.
 * @return Status code.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer, float *scale);

/**
 * Sets learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale New scaling factor.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitiv_Optimizer_set_learning_rate_scaling(
    primitiv_Optimizer *optimizer, float scale);

/**
 * Retrieves current L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength Pointer to receive the current L2 decay strength.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_weight_decay(
    const primitiv_Optimizer *optimizer, float *strength);

/**
 * Sets L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength New L2 decay strength, or 0 to disable L2 decay.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_weight_decay(
    primitiv_Optimizer *optimizer, float strength);

/**
 * Retrieves current gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold Current gradient clipping threshold.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer, float *threshold);

/**
 * Sets gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold New clipping threshold, or 0 to disable gradient clipping.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_set_gradient_clipping(
    primitiv_Optimizer *optimizer, float threshold);

/**
 * Registers a parameter.
 * @param optimizer Pointer of a handler.
 * @param param Parameter to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_parameter(
    primitiv_Optimizer *optimizer, primitiv_Parameter *param);

/**
 * Registers multiple parameters.
 * @param optimizer Pointer of a handler.
 * @param params List of parameters to be optimized.
 * @param n Number of parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_parameters(
  primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n);

/**
 * Registers a model.
 * @param optimizer Pointer of a handler.
 * @param param Model to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_model(
    primitiv_Optimizer *optimizer, primitiv_Model *model);

/**
 * Registers multiple models.
 * @param optimizer Pointer of a handler.
 * @param params List of models to be optimized.
 * @param n Number of models.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_add_models(
    primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n);

/**
 * Resets all gradients of registered parameters.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_reset_gradients(
  primitiv_Optimizer *optimizer);

/**
 * Updates parameter values.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Optimizer_update(
    primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_Optimizer configs

#endif  // PRIMITIV_C_OPTIMIZER_H_
