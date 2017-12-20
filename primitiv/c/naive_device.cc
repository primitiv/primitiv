/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/naive_device.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;

#define CAST_TO_CC_NAIVE(x) reinterpret_cast<Naive*>(x)
#define CAST_TO_CONST_CC_NAIVE(x) reinterpret_cast<const Naive*>(x)

extern "C" {

primitiv_Device *primitiv_Naive_new() {
  return to_c(new Naive());
}
primitiv_Device *safe_primitiv_Naive_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Naive_new(), status, nullptr);
}

primitiv_Device *primitiv_Naive_new_with_seed(uint32_t seed) {
  return to_c(new Naive(seed));
}
primitiv_Device *safe_primitiv_Naive_new_with_seed(uint32_t seed,
                                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Naive_new_with_seed(seed), status, nullptr);
}

void primitiv_Naive_delete(primitiv_Device *device) {
  delete CAST_TO_CC_NAIVE(device);
}
void safe_primitiv_Naive_delete(primitiv_Device *device,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Naive_delete(device), status);
}

void primitiv_Naive_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_NAIVE(device)->dump_description();
}
void safe_primitiv_Naive_dump_description(const primitiv_Device *device,
                                          primitiv_Status *status) {
  SAFE_EXPR(primitiv_Naive_dump_description(device), status);
}

}  // end extern "C"
