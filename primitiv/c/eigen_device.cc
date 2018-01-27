#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/eigen_device.h>

using primitiv::devices::Eigen;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitiv_devices_Eigen_new(primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new Eigen());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_devices_Eigen_new_with_seed(
    uint32_t rng_seed, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new Eigen(rng_seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
