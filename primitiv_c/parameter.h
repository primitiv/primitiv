#ifndef PRIMITIV_C_PARAMETER_H_
#define PRIMITIV_C_PARAMETER_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/shape.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Parameter primitiv_Parameter;

primitiv_Parameter *primitiv_Parameter_new();
primitiv_Parameter *safe_primitiv_Parameter_new(primitiv_Status *status);

primitiv_Parameter *primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device);
primitiv_Parameter *safe_primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status);

primitiv_Parameter *primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device);
primitiv_Parameter *safe_primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status);

void primitiv_Parameter_delete(primitiv_Parameter *parameter);
void safe_primitiv_Parameter_delete(primitiv_Parameter *parameter, primitiv_Status *status);

void primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device);
void safe_primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status);

void primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device);
void safe_primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status);

void primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device);
void safe_primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device,
    primitiv_Status *status);

void primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats);
void safe_primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Status *status);

bool primitiv_Parameter_valid(const primitiv_Parameter *parameter);
bool safe_primitiv_Parameter_valid(const primitiv_Parameter *parameter, primitiv_Status *status);

void primitiv_Parameter_reset_gradients(primitiv_Parameter *parameter);
void safe_primitiv_Parameter_reset_gradients(primitiv_Parameter *parameter, primitiv_Status *status);

void primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape);
void safe_primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape,
    primitiv_Status *status);

bool primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name);
bool safe_primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name,
    primitiv_Status *status);

primitiv_Shape *primitiv_Parameter_shape(const primitiv_Parameter *parameter);
primitiv_Shape *safe_primitiv_Parameter_shape(const primitiv_Parameter *parameter, primitiv_Status *status);

primitiv_Device *primitiv_Parameter_device(const primitiv_Parameter *parameter);
primitiv_Device *safe_primitiv_Parameter_device(const primitiv_Parameter *parameter, primitiv_Status *status);

const primitiv_Tensor *primitiv_Parameter_value(const primitiv_Parameter *parameter);
const primitiv_Tensor *safe_primitiv_Parameter_value(const primitiv_Parameter *parameter, primitiv_Status *status);

const primitiv_Tensor *primitiv_Parameter_gradient(const primitiv_Parameter *parameter);
const primitiv_Tensor *safe_primitiv_Parameter_gradient(const primitiv_Parameter *parameter, primitiv_Status *status);

const primitiv_Tensor *primitiv_Parameter_stats(const primitiv_Parameter *parameter, const char *name);
const primitiv_Tensor *safe_primitiv_Parameter_stats(const primitiv_Parameter *parameter, const char *name, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_PARAMETER_H_
