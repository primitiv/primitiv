/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include <primitiv/c/define.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum primitiv_Code {
  PRIMITIV_OK = 0,
  PRIMITIV_ERROR = 1,
  PRIMITIV_CANCELLED = 2,
  PRIMITIV_UNKNOWN = 3,
  PRIMITIV_INVALID_ARGUMENT = 4,
  PRIMITIV_DEADLINE_EXCEEDED = 5,
  PRIMITIV_NOT_FOUND = 6,
  PRIMITIV_ALREADY_EXISTS = 7,
  PRIMITIV_PERMISSION_DENIED = 8,
  PRIMITIV_RESOURCE_EXHAUSTED = 9,
  PRIMITIV_FAILED_PRECONDITION = 10,
  PRIMITIV_ABORTED = 11,
  PRIMITIV_OUT_OF_RANGE = 12,
  PRIMITIV_UNIMPLEMENTED = 13,
  PRIMITIV_INTERNAL = 14,
  PRIMITIV_UNAVAILABLE = 15,
  PRIMITIV_DATA_LOSS = 16,
  PRIMITIV_UNAUTHENTICATED = 17,
} primitiv_Code;

typedef struct primitiv_Status primitiv_Status;

CAPI extern primitiv_Status *primitiv_Status_new();

CAPI extern void primitiv_Status_delete(primitiv_Status *status);

CAPI extern void primitiv_Status_set_status(primitiv_Status *status,
                                            primitiv_Code code,
                                            const char *file,
                                            uint32_t line,
                                            const char *message);

CAPI extern primitiv_Code primitiv_Status_get_code(
    const primitiv_Status *status);

CAPI extern const char *primitiv_Status_get_message(
    const primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_STATUS_H_
