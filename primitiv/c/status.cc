/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/error.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>

using primitiv::Error;

extern "C" {

struct primitiv_Status {
  primitiv_Code code;
  primitiv::Error* error;
};

}  // end extern "C"

namespace primitiv {

void set_status(primitiv_Status *status,
                primitiv_Code code,
                const Error *error) {
  status->code = code;
  if (error) {
    status->error = new Error(*error);
  } else {
    status->error = nullptr;
  }
}

}  // namespace primitiv

extern "C" {

primitiv_Status *primitiv_Status_new() {
  return new primitiv_Status{PRIMITIV_OK, nullptr};
}

void primitiv_Status_delete(primitiv_Status *status) {
  delete status->error;
  delete status;
}

void primitiv_Status_set_status(primitiv_Status *status,
                                primitiv_Code code,
                                const char *file,
                                uint32_t line,
                                const char *message) {
  status->code = code;
  status->error = new Error(file, line, message);
}

primitiv_Code primitiv_Status_get_code(const primitiv_Status *status) {
  return status->code;
}

const char *primitiv_Status_get_message(const primitiv_Status *status) {
  if (status->error) {
    return status->error->what();
  } else {
    static const char *message = "";
    return message;
  }
}

}  // end extern "C"
