/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/device.h>

using primitiv::Device;
using primitiv::c::internal::to_c;
using primitiv::c::internal::to_cc;

extern "C" {

primitiv_Status primitiv_Device_get_default(primitiv_Device **device) {
  try {
    *device = to_c(&Device::get_default());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Device_set_default(primitiv_Device *device) {
  Device::set_default(*to_cc(device));
}

}  // end extern "C"
