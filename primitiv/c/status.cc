/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>

using primitiv::c::internal::ErrorHandler;

extern "C" {

primitiv_Status primitiv_Status_get_message(
    char *buffer, size_t *buffer_size) try {
  const std::string str = ErrorHandler::get_instance().get_message();
  if (buffer_size) {
    *buffer_size = str.length();
  }
  if (buffer) {
    std::strcpy(buffer, str.c_str());
  }
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Status_reset() try {
  ErrorHandler::get_instance().reset();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
