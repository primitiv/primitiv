/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_UTILS_H_
#define PRIMITIV_C_UTILS_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_StrIntMap primitiv_StrIntMap;

typedef struct primitiv_StrFloatMap primitiv_StrFloatMap;

CAPI extern primitiv_StrIntMap *primitiv_StrIntMap_new();
CAPI extern primitiv_StrIntMap *safe_primitiv_StrIntMap_new(
    primitiv_Status *status);

CAPI extern void primitiv_StrIntMap_delete(primitiv_StrIntMap *map);
CAPI extern void safe_primitiv_StrIntMap_delete(primitiv_StrIntMap *map,
                                                primitiv_Status *status);

CAPI extern void primitiv_StrIntMap_put(primitiv_StrIntMap *map,
                                        const char *key,
                                        uint32_t value);
CAPI extern void safe_primitiv_StrIntMap_put(primitiv_StrIntMap *map,
                                             const char *key,
                                             uint32_t value,
                                             primitiv_Status *status);

CAPI extern uint32_t primitiv_StrIntMap_get(const primitiv_StrIntMap *map,
                                            const char *key);
CAPI extern uint32_t safe_primitiv_StrIntMap_get(const primitiv_StrIntMap *map,
                                                 const char *key,
                                                 primitiv_Status *status);

CAPI extern void primitiv_StrIntMap_remove(primitiv_StrIntMap *map,
                                           const char *key);
CAPI extern void safe_primitiv_StrIntMap_remove(primitiv_StrIntMap *map,
                                                const char *key,
                                                primitiv_Status *status);

CAPI extern bool primitiv_StrIntMap_has(const primitiv_StrIntMap *map,
                                        const char *key);
CAPI extern bool safe_primitiv_StrIntMap_has(const primitiv_StrIntMap *map,
                                             const char *key,
                                             primitiv_Status *status);

CAPI extern const char *const *primitiv_StrIntMap_keys(primitiv_StrIntMap *map);
CAPI extern const char *const *safe_primitiv_StrIntMap_keys(
    primitiv_StrIntMap *map, primitiv_Status *status);

CAPI extern const uint32_t *primitiv_StrIntMap_values(primitiv_StrIntMap *map);
CAPI extern const uint32_t *safe_primitiv_StrIntMap_values(
    primitiv_StrIntMap *map, primitiv_Status *status);

CAPI extern void primitiv_StrIntMap_clear(primitiv_StrIntMap *map);
CAPI extern void safe_primitiv_StrIntMap_clear(
    primitiv_StrIntMap *map, primitiv_Status *status);

CAPI extern primitiv_StrFloatMap *primitiv_StrFloatMap_new();
CAPI extern primitiv_StrFloatMap *safe_primitiv_StrFloatMap_new(
    primitiv_Status *status);

CAPI extern void primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map);
CAPI extern void safe_primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map,
                                                  primitiv_Status *status);

CAPI extern void primitiv_StrFloatMap_put(primitiv_StrFloatMap *map,
                                          const char *key,
                                          float value);
CAPI extern void safe_primitiv_StrFloatMap_put(primitiv_StrFloatMap *map,
                                               const char *key,
                                               float value,
                                               primitiv_Status *status);

CAPI extern float primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map,
                                           const char *key);
CAPI extern float safe_primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map,
                                                const char *key,
                                                primitiv_Status *status);

CAPI extern void primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map,
                                             const char *key);
CAPI extern void safe_primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map,
                                                  const char *key,
                                                  primitiv_Status *status);

CAPI extern bool primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map,
                                          const char *key);
CAPI extern bool safe_primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map,
                                               const char *key,
                                               primitiv_Status *status);

CAPI extern const char *const *primitiv_StrFloatMap_keys(
    primitiv_StrFloatMap *map);
CAPI extern const char *const *safe_primitiv_StrFloatMap_keys(
    primitiv_StrFloatMap *map, primitiv_Status *status);

CAPI extern const float *primitiv_StrFloatMap_values(primitiv_StrFloatMap *map);
CAPI extern const float *safe_primitiv_StrFloatMap_values(
    primitiv_StrFloatMap *map, primitiv_Status *status);

CAPI extern void primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map);
CAPI extern void safe_primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map,
                                                 primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_UTILS_H_
