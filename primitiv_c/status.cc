#include <primitiv/error.h>

#include <sstream>

#include "primitiv_c/internal.h"
#include "primitiv_c/status.h"

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
                const Error &error) {
  status->code = code;
  status->error = new Error(error);
}

}  // namespace primitiv

extern "C" {

primitiv_Status *primitiv_Status_new() {
  return new primitiv_Status{PRIMITIV_UNKNOWN, nullptr};
}

void primitiv_Status_delete(primitiv_Status *status) {
  if (status->error) {
    delete status->error;
  }
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
    std::stringstream ss;
    ss << "Status has no error: initial status(" << status->code << ").";
    static const char *message = ss.str().c_str();
    return message;
  }
}

}  // end extern "C"
