#include "primitiv_c/internal.h"
#include "primitiv_c/naive_device.h"

#include <primitiv/naive_device.h>

using primitiv::devices::Naive;

extern "C" {

primitiv_Device* primitiv_Naive_new() {
  return reinterpret_cast<primitiv_Device*>(new Naive());
}

primitiv_Device* primitiv_Naive_new_with_seed(uint32_t seed) {
  return reinterpret_cast<primitiv_Device*>(new Naive(seed));
}

void primitiv_Naive_delete(primitiv_Device *device) {
  delete reinterpret_cast<Naive*>(device);
}

}  // end extern "C"
