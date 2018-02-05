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
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteOptimizer(
    primitivOptimizer_t *optimizer);

/**
 * Loads configurations from a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the optimizer parameter file.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivLoadOptimizer(
    primitivOptimizer_t *optimizer, const char *path);

/**
 * Saves current configurations to a file.
 * @param optimizer Pointer of a handler.
 * @param path Path of the file that will store optimizer parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSaveOptimizer(
    const primitivOptimizer_t *optimizer, const char *path);

/**
 * Retrieves current epoch.
 * @param optimizer Pointer of a handler.
 * @param retcval Pointer to receive the current epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetOptimizerEpoch(
    const primitivOptimizer_t *optimizer, uint32_t *retval);

/**
 * Sets current epoch.
 * @param optimizer Pointer of a handler.
 * @param epoch New epoch.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetOptimizerEpoch(
    primitivOptimizer_t *optimizer, uint32_t epoch);

/**
 * Retrieves current learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param retval Pointer to receive the scaling factor.
 * @return Status code.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitivGetOptimizerLearningRateScaling(
    const primitivOptimizer_t *optimizer, float *retval);

/**
 * Sets learning rate scaling factor.
 * @param optimizer Pointer of a handler.
 * @param scale New scaling factor.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API
PRIMITIV_C_STATUS primitivSetOptimizerLearningRateScaling(
    primitivOptimizer_t *optimizer, float scale);

/**
 * Retrieves current L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param retval Pointer to receive the current L2 decay strength.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetOptimizerWeightDecay(
    const primitivOptimizer_t *optimizer, float *retval);

/**
 * Sets L2 decay strength.
 * @param optimizer Pointer of a handler.
 * @param strength New L2 decay strength, or 0 to disable L2 decay.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetOptimizerWeightDecay(
    primitivOptimizer_t *optimizer, float strength);

/**
 * Retrieves current gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param retval Current gradient clipping threshold.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetOptimizerGradientClipping(
    const primitivOptimizer_t *optimizer, float *retval);

/**
 * Sets gradient clipping threshold.
 * @param optimizer Pointer of a handler.
 * @param threshold New clipping threshold, or 0 to disable gradient clipping.
 * @return Status code.
 * @remarks Could not set negative values.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetOptimizerGradientClipping(
    primitivOptimizer_t *optimizer, float threshold);

/**
 * Registers a parameter.
 * @param optimizer Pointer of a handler.
 * @param param Parameter to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivAddParameterToOptimizer(
    primitivOptimizer_t *optimizer, primitivParameter_t *param);

/**
 * Registers multiple parameters.
 * @param optimizer Pointer of a handler.
 * @param params List of parameters to be optimized.
 * @param n Number of parameters.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivAddParametersToOptimizer(
  primitivOptimizer_t *optimizer, primitivParameter_t **params, size_t n);

/**
 * Registers a model.
 * @param optimizer Pointer of a handler.
 * @param param Model to be optimized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivAddModelToOptimizer(
    primitivOptimizer_t *optimizer, primitivModel_t *model);

/**
 * Registers multiple models.
 * @param optimizer Pointer of a handler.
 * @param params List of models to be optimized.
 * @param n Number of models.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivAddModelsToOptimizer(
    primitivOptimizer_t *optimizer, primitivModel_t **models, size_t n);

/**
 * Resets all gradients of registered parameters.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivResetOptimizerGradients(
  primitivOptimizer_t *optimizer);

/**
 * Updates parameter values.
 * @param optimizer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivExecuteOptimizerUpdate(
    primitivOptimizer_t *optimizer);

/**
 * Gets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param retval Pointer to receive the value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetOptimizerIntConfig(
    const primitivOptimizer_t *optimizer, const char *key, uint32_t *retval);

/**
 * Sets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Configuration value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetOptimizerIntConfig(
    primitivOptimizer_t *optimizer, const char *key, uint32_t value);

/**
 * Gets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param retval Pointer to receive the value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetOptimizerFloatConfig(
    const primitivOptimizer_t *optimizer, const char *key, float *retval);

/**
 * Sets a configuration value.
 * @param optimizer Pointer of a handler.
 * @param key Configuration name.
 * @param value Configuration value.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetOptimizerFloatConfig(
    primitivOptimizer_t *optimizer, const char *key, float value);

#endif  // PRIMITIV_C_OPTIMIZER_H_
