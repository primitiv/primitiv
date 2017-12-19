/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include <primitiv/c/define.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum primitiv_Status {
  PRIMITIV_OK = 0,
  PRIMITIV_ERROR = 1,
} primitiv_Status;

CAPI extern const char *primitiv_Status_get_message();

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_STATUS_H_
