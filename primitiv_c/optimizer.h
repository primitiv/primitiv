#ifndef PRIMITIV_C_OPTIMIZER_H_
#define PRIMITIV_C_OPTIMIZER_H_

#include "primitiv_c/define.h"
#include "primitiv_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Optimizer primitiv_Optimizer;

void primitiv_Optimizer_load(primitiv_Optimizer *optimizer, const char *path);
void safe_primitiv_Optimizer_load(primitiv_Optimizer *optimizer, const char *path, primitiv_Status *status);

void primitiv_Optimizer_save(const primitiv_Optimizer *optimizer, const char *path);
void safe_primitiv_Optimizer_save(const primitiv_Optimizer *optimizer, const char *path, primitiv_Status *status);

uint32_t primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer);
uint32_t safe_primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer, primitiv_Status *status);

void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer, uint32_t epoch);
void safe_primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer, uint32_t epoch, primitiv_Status *status);

float primitiv_Optimizer_get_learning_rate_scaling(const primitiv_Optimizer *optimizer);
float safe_primitiv_Optimizer_get_learning_rate_scaling(const primitiv_Optimizer *optimizer, primitiv_Status *status);

void primitiv_Optimizer_set_learning_rate_scaling(primitiv_Optimizer *optimizer, float scale);
void safe_primitiv_Optimizer_set_learning_rate_scaling(primitiv_Optimizer *optimizer, float scale, primitiv_Status *status);

float primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer);
float safe_primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer, primitiv_Status *status);

void primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer, float strength);
void safe_primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer, float strength, primitiv_Status *status);

float primitiv_Optimizer_get_gradient_clipping(const primitiv_Optimizer *optimizer);
float safe_primitiv_Optimizer_get_gradient_clipping(const primitiv_Optimizer *optimizer, primitiv_Status *status);

void primitiv_Optimizer_set_gradient_clipping(primitiv_Optimizer *optimizer, float threshold);
void safe_primitiv_Optimizer_set_gradient_clipping(primitiv_Optimizer *optimizer, float threshold, primitiv_Status *status);

void primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer, primitiv_Parameter *param);
void safe_primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer, primitiv_Parameter *param, primitiv_Status *status);

void primitiv_Optimizer_add_parameters(primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n);
void safe_primitiv_Optimizer_add_parameters(primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n, primitiv_Status *status);

void primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer, primitiv_Model *model);
void safe_primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer, primitiv_Model *model, primitiv_Status *status);

void primitiv_Optimizer_add_models(primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n);
void safe_primitiv_Optimizer_add_models(primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n, primitiv_Status *status);

void primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer);
void safe_primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer, primitiv_Status *status);

void primitiv_Optimizer_update(primitiv_Optimizer *optimizer);
void safe_primitiv_Optimizer_update(primitiv_Optimizer *optimizer, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_H_
