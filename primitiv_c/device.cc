#include "primitiv_c/device.h"

#include <primitiv/device.h>
#include <primitiv/naive_device.h>

using primitiv::Device;
using primitiv::devices::Naive;

extern "C" {

struct primitiv_Device {
  Device *device;
};

primitiv_Device *primitiv_Device_get_default() {
  return new primitiv_Device{&Device::get_default()};
}

void primitiv_Device_set_default(primitiv_Device *device) {
  Device::set_default(*device->device);
}

void primitiv_Device_delete(primitiv_Device *device) {
  delete device->device;
  delete device;
}

primitiv_Device* primitiv_Naive_new() {
  return new primitiv_Device{new Naive()};
}

primitiv_Device* primitiv_Naive_new_with_seed(uint32_t seed) {
  return new primitiv_Device{new Naive(seed)};
}

/*
struct primitiv_Naive {
  primitiv::devices::Naive device;
};

primitiv_Naive* primitiv_Naive_new() {
  return new primitiv_Naive;
}

primitiv_Naive* primitiv_Naive_new_with_seed(uint32_t seed) {
  return new primitiv_Naive(seed);
}

void primitiv_Naive_delete(primitiv_Naive* device) {
  delete device;
}
*/

}  // end extern "C"
