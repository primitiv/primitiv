/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <string>

#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>

using primitiv::c::internal::ErrorHandler;

extern "C" {

primitiv_Status primitiv_Status_get_message(
    char *buffer, size_t *buffer_size) try {
  PRIMITIV_C_CHECK_PTR_ARG(buffer_size);
  primitiv::c::internal::copy_string_to_array(
      ErrorHandler::get_instance().get_message(), buffer, buffer_size);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Status_reset() try {
  ErrorHandler::get_instance().reset();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
