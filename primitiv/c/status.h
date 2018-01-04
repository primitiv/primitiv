#ifndef PRIMITIV_C_STATUS_H_
#define PRIMITIV_C_STATUS_H_

#include <primitiv/c/define.h>

PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_reset();

PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_get_message(
    char *buffer, size_t *buffer_size);

#endif  // PRIMITIV_C_STATUS_H_
