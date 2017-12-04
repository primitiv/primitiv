#include "primitiv_c/internal.h"
#include "primitiv_c/naive_device.h"

#include <primitiv/naive_device.h>

using primitiv::devices::Naive;

#define CAST_TO_CC_NAIVE(x) reinterpret_cast<Naive*>(x)
#define CAST_TO_CONST_CC_NAIVE(x) reinterpret_cast<const Naive*>(x)

extern "C" {

primitiv_Device *primitiv_Naive_new() {
  return to_c(new Naive());
}

primitiv_Device *primitiv_Naive_new_with_seed(uint32_t seed) {
  return to_c(new Naive(seed));
}

void primitiv_Naive_delete(primitiv_Device *device) {
  delete CAST_TO_CC_NAIVE(device);
}

}  // end extern "C"
