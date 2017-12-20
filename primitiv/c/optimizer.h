/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_OPTIMIZER_H_
#define PRIMITIV_C_OPTIMIZER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque type of Optimizer.
 */
typedef struct primitiv_Optimizer primitiv_Optimizer;

/**
 * Deletes the Optimizer object.
 * @param optimizer Pointer of a handler.
 */
CAPI extern void primitiv_Optimizer_delete(primitiv_Optimizer *optimizer);

/**
 * Loads configurations from a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the optimizer parameter file.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_load(
    primitiv_Optimizer *optimizer, const char *path);

/**
 * Saves current configurations to a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the file that will store optimizer parameters.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_save(
    const primitiv_Optimizer *optimizer, const char *path);

/**
 * Retrieves current epoch.
 * @param optimizer Pointer of a handler.
 * @return Current epoch.
 */
CAPI extern uint32_t primitiv_Optimizer_get_epoch(
    const primitiv_Optimizer *optimizer);

/**
 * Sets current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch New epoch.
 */
CAPI extern void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer,
                                              uint32_t epoch);

/**
 * Retrieves current learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @return The scaling factor.
 */
CAPI extern float primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer);

/**
 * Sets learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale New scaling factor.
 * @return Status code.
 * @remarks Could not set negative values.
 */
CAPI extern primitiv_Status primitiv_Optimizer_set_learning_rate_scaling(
    primitiv_Optimizer *optimizer, float scale);

/**
 * Retrieves current L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @return Current L2 decay strength.
 */
CAPI extern float primitiv_Optimizer_get_weight_decay(
    const primitiv_Optimizer *optimizer);

/**
 * Sets L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength New L2 decay strength, or 0 to disable L2 decay.
 * @return Status code.
 * @remarks Could not set negative values.
 */
CAPI extern primitiv_Status primitiv_Optimizer_set_weight_decay(
    primitiv_Optimizer *optimizer, float strength);

/**
 * Retrieves current gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @return Current gradient clipping threshold.
 */
CAPI extern float primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer);

/**
 * Sets gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold New clipping threshold, or 0 to disable gradient clipping.
 * @return Status code.
 * @remarks Could not set negative values.
 */
CAPI extern primitiv_Status primitiv_Optimizer_set_gradient_clipping(
    primitiv_Optimizer *optimizer, float threshold);

/**
 * Registers a parameter.
 * @param optimizer Pointer of a handler.
 * @param param Parameter to be optimized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_add_parameter(
    primitiv_Optimizer *optimizer, primitiv_Parameter *param);

/**
 * Registers multiple parameters.
 * @param optimizer Pointer of a handler.
 * @param params List of parameters to be optimized.
 * @param n Number of parameters.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_add_parameters(
  primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n);

/**
 * Registers a model.
 * @param optimizer Pointer of a handler.
 * @param param Model to be optimized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_add_model(
    primitiv_Optimizer *optimizer, primitiv_Model *model);

/**
 * Registers multiple models.
 * @param optimizer Pointer of a handler.
 * @param params List of models to be optimized.
 * @param n Number of models.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_add_models(
    primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n);

/**
 * Resets all gradients of registered parameters.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_reset_gradients(
  primitiv_Optimizer *optimizer);

/**
 * Updates parameter values.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Optimizer_update(
    primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_Optimizer configs

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_H_
