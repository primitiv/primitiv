/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/device.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/device.h>

using primitiv::Device;

extern "C" {

primitiv_Device *primitiv_Device_get_default() {
  return to_c(&Device::get_default());
}
primitiv_Device *safe_primitiv_Device_get_default(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Device_get_default(), status, nullptr);
}

void primitiv_Device_set_default(primitiv_Device *device) {
  Device::set_default(*to_cc(device));
}
void safe_primitiv_Device_set_default(primitiv_Device *device,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_Device_set_default(device), status);
}

}  // end extern "C"
