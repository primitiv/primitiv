/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/naive_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

#define CAST_TO_CC_NAIVE(x) reinterpret_cast<Naive*>(x)
#define CAST_TO_CONST_CC_NAIVE(x) reinterpret_cast<const Naive*>(x)

extern "C" {

primitiv_Device *primitiv_devices_Naive_new() {
  return to_c_ptr(new Naive());
}

primitiv_Device *primitiv_devices_Naive_new_with_seed(uint32_t seed) {
  return to_c_ptr(new Naive(seed));
}

void primitiv_devices_Naive_delete(primitiv_Device *device) {
  delete CAST_TO_CC_NAIVE(device);
}

void primitiv_devices_Naive_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_NAIVE(device)->dump_description();
}

}  // end extern "C"

#undef CAST_TO_CC_NAIVE
#undef CAST_TO_CONST_CC_NAIVE
