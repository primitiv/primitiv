#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include "primitiv_c/define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum primitiv_Code {
  PRIMITIV_OK = 0,
  PRIMITIV_CANCELLED = 1,
  PRIMITIV_UNKNOWN = 2,
  PRIMITIV_INVALID_ARGUMENT = 3,
  PRIMITIV_DEADLINE_EXCEEDED = 4,
  PRIMITIV_NOT_FOUND = 5,
  PRIMITIV_ALREADY_EXISTS = 6,
  PRIMITIV_PERMISSION_DENIED = 7,
  PRIMITIV_RESOURCE_EXHAUSTED = 8,
  PRIMITIV_FAILED_PRECONDITION = 9,
  PRIMITIV_ABORTED = 10,
  PRIMITIV_OUT_OF_RANGE = 11,
  PRIMITIV_UNIMPLEMENTED = 12,
  PRIMITIV_INTERNAL = 13,
  PRIMITIV_UNAVAILABLE = 14,
  PRIMITIV_DATA_LOSS = 15,
  PRIMITIV_UNAUTHENTICATED = 16,
} primitiv_Code;

typedef struct primitiv_Status primitiv_Status;

primitiv_Status *primitiv_Status_new();

void primitiv_Status_delete(primitiv_Status *status);

void primitiv_Status_set_status(primitiv_Status *status, primitiv_Code code, const char *file, uint32_t line, const char *message);

primitiv_Code primitiv_Status_get_code(const primitiv_Status *status);

const char *primitiv_Status_get_message(const primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_STATUS_H_
