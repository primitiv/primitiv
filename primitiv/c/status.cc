#include <primitiv/config.h>

#include <string>

#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>

using primitiv::c::internal::ErrorHandler;

primitiv_Status primitiv_Status_get_message(
    char *buffer, size_t *buffer_size) try {
  PRIMITIV_C_CHECK_NOT_NULL(buffer_size);
  primitiv::c::internal::copy_string_to_array(
      ErrorHandler::get_instance().get_message(), buffer, buffer_size);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Status_reset() try {
  ErrorHandler::get_instance().reset();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
