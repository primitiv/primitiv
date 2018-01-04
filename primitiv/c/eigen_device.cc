/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/eigen_device.h>

using primitiv::devices::Eigen;
using primitiv::c::internal::to_c_ptr;

primitiv_Status primitiv_devices_Eigen_new(primitiv_Device **device) try {
  *device = to_c_ptr(new Eigen());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_Eigen_new_with_seed(
    uint32_t rng_seed, primitiv_Device **device) try {
  *device = to_c_ptr(new Eigen(rng_seed));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
