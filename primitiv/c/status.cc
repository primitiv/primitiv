/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>

using primitiv_c::internal::ErrorHandler;

extern "C" {

const char *primitiv_Status_get_message() {
  return ErrorHandler::get_instance().get_message();
}

}  // end extern "C"
