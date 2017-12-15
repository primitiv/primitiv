/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_MODEL_H_
#define PRIMITIV_C_MODEL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Model primitiv_Model;

CAPI extern primitiv_Model *primitiv_Model_new();
CAPI extern primitiv_Model *safe_primitiv_Model_new(primitiv_Status *status);

CAPI extern void primitiv_Model_delete(primitiv_Model *model);
CAPI extern void safe_primitiv_Model_delete(primitiv_Model *model,
                                            primitiv_Status *status);

CAPI extern void primitiv_Model_load(primitiv_Model *model,
                                     const char *path,
                                     bool with_stats,
                                     primitiv_Device *device);
CAPI extern void safe_primitiv_Model_load(primitiv_Model *model,
                                          const char *path,
                                          bool with_stats,
                                          primitiv_Device *device,
                                          primitiv_Status *status);

CAPI extern void primitiv_Model_save(const primitiv_Model *model,
                                     const char *path,
                                     bool with_stats);
CAPI extern void safe_primitiv_Model_save(const primitiv_Model *model,
                                          const char *path,
                                          bool with_stats,
                                          primitiv_Status *status);

CAPI extern void primitiv_Model_add_parameter(primitiv_Model *model,
                                              const char *name,
                                              primitiv_Parameter *param);
CAPI extern void safe_primitiv_Model_add_parameter(primitiv_Model *model,
                                                   const char *name,
                                                   primitiv_Parameter *param,
                                                   primitiv_Status *status);

CAPI extern void primitiv_Model_add_model(primitiv_Model *self,
                                          const char *name,
                                          primitiv_Model *param);
CAPI extern void safe_primitiv_Model_add_model(primitiv_Model *self,
                                               const char *name,
                                               primitiv_Model *model,
                                               primitiv_Status *status);

CAPI extern const primitiv_Parameter *primitiv_Model_get_parameter(
    const primitiv_Model *model,
    const char **names,
    size_t n);
CAPI extern const primitiv_Parameter *safe_primitiv_Model_get_parameter(
    const primitiv_Model *model,
    const char **names,
    size_t n,
    primitiv_Status *status);

CAPI extern const primitiv_Model *primitiv_Model_get_submodel(
    const primitiv_Model *model,
    const char **names,
    size_t n);
CAPI extern const primitiv_Model *safe_primitiv_Model_get_submodel(
    const primitiv_Model *model,
    const char **names,
    size_t n,
    primitiv_Status *status);

// @TODO: Implement primitiv_Model_get_all_parameters()
// @TODO: Implement primitiv_Model_get_trainable_parameters()

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_MODEL_H_
