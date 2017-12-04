#include "primitiv_c/internal.h"
#include "primitiv_c/device.h"

#include <primitiv/device.h>

using primitiv::Device;

extern "C" {

primitiv_Device *primitiv_Device_get_default() {
  return  CAST_TO_C_DEVICE(&Device::get_default());
}

void primitiv_Device_set_default(primitiv_Device *device) {
  Device::set_default(*CAST_TO_CC_DEVICE(device));
}

}  // end extern "C"
