#ifndef PRIMITIV_C_OPTIMIZER_H_
#define PRIMITIV_C_OPTIMIZER_H_

#include "primitiv_c/define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Optimizer primitiv_Optimizer;

primitiv_Optimizer *primitiv_Optimizer_new();

void primitiv_Optimizer_delete(primitiv_Optimizer *optimizer);

void primitiv_Optimizer_load(primitiv_Optimizer *optimizer, const char *path);

void primitiv_Optimizer_save(const primitiv_Optimizer *optimizer, const char *path);

uint32_t primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer);

void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer, uint32_t epoch);

float primitiv_Optimizer_get_learning_rate_scaling(const primitiv_Optimizer *optimizer);

void primitiv_Optimizer_set_learning_rate_scaling(primitiv_Optimizer *optimizer, float scale);

float primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer);

void primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer, float strength);

float primitiv_Optimizer_get_gradient_clipping(const primitiv_Optimizer *optimizer);

void primitiv_Optimizer_set_gradient_clipping(primitiv_Optimizer *optimizer, float threshold);

void primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer, primitiv_Parameter *param);

void primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer, const primitiv_Model *model);

void primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer);

void primitiv_Optimizer_update(primitiv_Optimizer *optimizer);

// @TODO: implement get_configs();

// @TODO: implement set_configs();

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_H_
