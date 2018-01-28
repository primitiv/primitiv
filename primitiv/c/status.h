#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include <primitiv/c/define.h>

PRIMITIV_C_API PRIMITIV_C_STATUS primitivResetStatus();

PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetMessage(char *retval, size_t *size);

#endif  // PRIMITIV_C_STATUS_H_
