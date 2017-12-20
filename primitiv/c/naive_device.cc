/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/naive_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

extern "C" {

primitiv_Device *primitiv_devices_Naive_new() {
  return to_c_ptr(new Naive());
}

primitiv_Device *primitiv_devices_Naive_new_with_seed(uint32_t seed) {
  return to_c_ptr(new Naive(seed));
}

}  // end extern "C"
