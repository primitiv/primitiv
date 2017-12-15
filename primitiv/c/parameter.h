/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_PARAMETER_H_
#define PRIMITIV_C_PARAMETER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Parameter primitiv_Parameter;

CAPI extern primitiv_Parameter *primitiv_Parameter_new();
CAPI extern primitiv_Parameter *safe_primitiv_Parameter_new(
    primitiv_Status *status);

CAPI extern primitiv_Parameter *primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device);
CAPI extern primitiv_Parameter *safe_primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status);

CAPI extern primitiv_Parameter *primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device);
CAPI extern primitiv_Parameter *safe_primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status);

CAPI extern void primitiv_Parameter_delete(primitiv_Parameter *parameter);
CAPI extern void safe_primitiv_Parameter_delete(primitiv_Parameter *parameter,
                                                primitiv_Status *status);

CAPI extern void primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device);
CAPI extern void safe_primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status);

CAPI extern void primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device);
CAPI extern void safe_primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status);

CAPI extern void primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device);
CAPI extern void safe_primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device,
    primitiv_Status *status);

CAPI extern void primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats);
CAPI extern void safe_primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Status *status);

CAPI extern bool primitiv_Parameter_valid(const primitiv_Parameter *parameter);
CAPI extern bool safe_primitiv_Parameter_valid(
    const primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern void primitiv_Parameter_reset_gradients(
    primitiv_Parameter *parameter);
CAPI extern void safe_primitiv_Parameter_reset_gradients(
    primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern void primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape);
CAPI extern void safe_primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape,
    primitiv_Status *status);

CAPI extern bool primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name);
CAPI extern bool safe_primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name,
    primitiv_Status *status);

CAPI extern primitiv_Shape *primitiv_Parameter_shape(
    const primitiv_Parameter *parameter);
CAPI extern primitiv_Shape *safe_primitiv_Parameter_shape(
    const primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_Parameter_device(
    const primitiv_Parameter *parameter);
CAPI extern primitiv_Device *safe_primitiv_Parameter_device(
    const primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern const primitiv_Tensor *primitiv_Parameter_value(
    const primitiv_Parameter *parameter);
CAPI extern const primitiv_Tensor *safe_primitiv_Parameter_value(
    const primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern const primitiv_Tensor *primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter);
CAPI extern const primitiv_Tensor *safe_primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter, primitiv_Status *status);

CAPI extern const primitiv_Tensor *primitiv_Parameter_stats(
    const primitiv_Parameter *parameter, const char *name);
CAPI extern const primitiv_Tensor *safe_primitiv_Parameter_stats(
    const primitiv_Parameter *parameter,
    const char *name,
    primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_PARAMETER_H_
