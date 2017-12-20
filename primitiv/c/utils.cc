/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/utils.h>

extern "C" {

primitiv_StrIntMap *primitiv_StrIntMap_new() {
  return to_c(new StrIntMap());
}
primitiv_StrIntMap *safe_primitiv_StrIntMap_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrIntMap_new(), status, nullptr);
}

void primitiv_StrIntMap_delete(primitiv_StrIntMap *map) {
  delete to_cc(map);
}
void safe_primitiv_StrIntMap_delete(primitiv_StrIntMap *map,
                                    primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrIntMap_delete(map), status);
}

void primitiv_StrIntMap_put(primitiv_StrIntMap *map,
                            const char *key,
                            uint32_t value) {
  (*to_cc(map))[key] = value;
}
void safe_primitiv_StrIntMap_put(primitiv_StrIntMap *map,
                                 const char *key,
                                 uint32_t value,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrIntMap_put(map, key, value), status);
}

uint32_t primitiv_StrIntMap_get(const primitiv_StrIntMap *map,
                                const char *key) {
  return to_cc(map)->at(key);
}
uint32_t safe_primitiv_StrIntMap_get(const primitiv_StrIntMap *map,
                                     const char *key,
                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrIntMap_get(map, key), status, 0);
}

void primitiv_StrIntMap_remove(primitiv_StrIntMap *map, const char *key) {
  to_cc(map)->erase(key);
}
void safe_primitiv_StrIntMap_remove(primitiv_StrIntMap *map,
                                    const char *key,
                                    primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrIntMap_remove(map, key), status);
}

bool primitiv_StrIntMap_has(const primitiv_StrIntMap *map, const char *key) {
  return to_cc(map)->count(key) > 0;
}
bool safe_primitiv_StrIntMap_has(const primitiv_StrIntMap *map,
                                 const char *key,
                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrIntMap_has(map, key), status, false);
}

const char *const *primitiv_StrIntMap_keys(primitiv_StrIntMap *map) {
  return &(to_cc(map)->keys())[0];
}
const char *const *safe_primitiv_StrIntMap_keys(primitiv_StrIntMap *map,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrIntMap_keys(map), status, nullptr);
}

const uint32_t *primitiv_StrIntMap_values(primitiv_StrIntMap *map) {
  return &(to_cc(map)->values())[0];
}
const uint32_t *safe_primitiv_StrIntMap_values(primitiv_StrIntMap *map,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrIntMap_values(map), status, nullptr);
}

void primitiv_StrIntMap_clear(primitiv_StrIntMap *map) {
  to_cc(map)->clear();
}
void safe_primitiv_StrIntMap_clear(primitiv_StrIntMap *map,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrIntMap_clear(map), status);
}

primitiv_StrFloatMap *primitiv_StrFloatMap_new() {
  return to_c(new StrFloatMap());
}
primitiv_StrFloatMap *safe_primitiv_StrFloatMap_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrFloatMap_new(), status, nullptr);
}

void primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map) {
  delete to_cc(map);
}
void safe_primitiv_StrFloatMap_delete(primitiv_StrFloatMap *map,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrFloatMap_delete(map), status);
}

void primitiv_StrFloatMap_put(primitiv_StrFloatMap *map,
                              const char *key,
                              float value) {
  (*to_cc(map))[key] = value;
}
void safe_primitiv_StrFloatMap_put(primitiv_StrFloatMap *map,
                                   const char *key,
                                   float value,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrFloatMap_put(map, key, value), status);
}

float primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map,
                               const char *key) {
  return to_cc(map)->at(key);
}
float safe_primitiv_StrFloatMap_get(const primitiv_StrFloatMap *map,
                                    const char *key,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrFloatMap_get(map, key), status, 0.0);
}

void primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map, const char *key) {
  to_cc(map)->erase(key);
}
void safe_primitiv_StrFloatMap_remove(primitiv_StrFloatMap *map,
                                      const char *key,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrFloatMap_remove(map, key), status);
}

bool primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map,
                              const char *key) {
  return to_cc(map)->count(key) > 0;
}
bool safe_primitiv_StrFloatMap_has(const primitiv_StrFloatMap *map,
                                   const char *key,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrFloatMap_has(map, key), status, false);
}

const char *const *primitiv_StrFloatMap_keys(primitiv_StrFloatMap *map) {
  return &(to_cc(map)->keys())[0];
}
const char *const *safe_primitiv_StrFloatMap_keys(primitiv_StrFloatMap *map,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrFloatMap_keys(map), status, nullptr);
}

const float *primitiv_StrFloatMap_values(primitiv_StrFloatMap *map) {
  return &(to_cc(map)->values())[0];
}
const float *safe_primitiv_StrFloatMap_values(primitiv_StrFloatMap *map,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_StrFloatMap_values(map), status, nullptr);
}

void primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map) {
  to_cc(map)->clear();
}
void safe_primitiv_StrFloatMap_clear(primitiv_StrFloatMap *map,
                                     primitiv_Status *status) {
  SAFE_EXPR(primitiv_StrFloatMap_clear(map), status);
}

}  // end extern "C"
