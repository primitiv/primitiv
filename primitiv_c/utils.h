#ifndef PRIMITIV_C_UTILS_H_
#define PRIMITIV_C_UTILS_H_

#include "primitiv_c/define.h"
#include "primitiv_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_StrIntMap primitiv_StrIntMap;

typedef struct primitiv_StrFloatMap primitiv_StrFloatMap;

primitiv_StrIntMap *primitiv_StrIntMap_new();
primitiv_StrIntMap *safe_primitiv_StrIntMap_new(primitiv_Status *status);

void primitiv_StrIntMap_delete(primitiv_StrIntMap *map);
void safe_primitiv_StrIntMap_delete(primitiv_StrIntMap *map, primitiv_Status *status);

void primitiv_StrIntMap_put(primitiv_StrIntMap *map, const char *key, uint32_t value);
void safe_primitiv_StrIntMap_put(primitiv_StrIntMap *map, const char *key, uint32_t value, primitiv_Status *status);

uint32_t primitiv_StrIntMap_get(const primitiv_StrIntMap *map, const char *key);
uint32_t safe_primitiv_StrIntMap_get(const primitiv_StrIntMap *map, const char *key, primitiv_Status *status);

void primitiv_StrIntMap_remove(primitiv_StrIntMap *map, const char *key);
void safe_primitiv_StrIntMap_remove(primitiv_StrIntMap *map, const char *key, primitiv_Status *status);

bool primitiv_StrIntMap_has(const primitiv_StrIntMap *map, const char *key);
bool safe_primitiv_StrIntMap_has(const primitiv_StrIntMap *map, const char *key, primitiv_Status *status);

const char *const *primitiv_StrIntMap_keys(primitiv_StrIntMap *map);
const char *const *safe_primitiv_StrIntMap_keys(primitiv_StrIntMap *map, primitiv_Status *status);

const uint32_t *primitiv_StrIntMap_values(primitiv_StrIntMap *map);
const uint32_t *safe_primitiv_StrIntMap_values(primitiv_StrIntMap *map, primitiv_Status *status);

void primitiv_StrIntMap_clear(primitiv_StrIntMap *map);
void safe_primitiv_StrIntMap_clear(primitiv_StrIntMap *map, primitiv_Status *status);

primitiv_StrFloatMap *primitiv_StrFloatMap_new();
primitiv_StrFloatMap *safe_primitiv_StrFloatMap_new(primitiv_Status *status);

void primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map);
void safe_primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map, primitiv_Status *status);

void primitiv_StrFloatMap_put(primitiv_StrFloatMap *map, const char *key, float value);
void safe_primitiv_StrFloatMap_put(primitiv_StrFloatMap *map, const char *key, float value, primitiv_Status *status);

float primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map, const char *key);
float safe_primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map, const char *key, primitiv_Status *status);

void primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map, const char *key);
void safe_primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map, const char *key, primitiv_Status *status);

bool primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map, const char *key);
bool safe_primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map, const char *key, primitiv_Status *status);

const char *const *primitiv_StrFloatMap_keys(primitiv_StrFloatMap *map);
const char *const *safe_primitiv_StrFloatMap_keys(primitiv_StrFloatMap *map, primitiv_Status *status);

const float *primitiv_StrFloatMap_values(primitiv_StrFloatMap *map);
const float *safe_primitiv_StrFloatMap_values(primitiv_StrFloatMap *map, primitiv_Status *status);

void primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map);
void safe_primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_UTILS_H_
