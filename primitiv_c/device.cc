#include "primitiv_c/device.h"

#include <primitiv/device.h>

using primitiv::Device;

extern "C" {

struct primitiv_Device;

primitiv_Device *primitiv_Device_get_default() {
  return reinterpret_cast<primitiv_Device*>(&Device::get_default());
}

void primitiv_Device_set_default(primitiv_Device *device) {
  Device::set_default(*reinterpret_cast<Device*>(device));
}

}  // end extern "C"
