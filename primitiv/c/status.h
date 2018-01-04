#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include <primitiv/c/define.h>

typedef enum primitiv_Status {
  PRIMITIV_OK = 0,
  PRIMITIV_ERROR = -1,
} primitiv_Status;

PRIMITIV_C_API primitiv_Status primitiv_Status_get_message(
    char *buffer, size_t *buffer_size);

PRIMITIV_C_API primitiv_Status primitiv_Status_reset();

#endif  // PRIMITIV_C_STATUS_H_
