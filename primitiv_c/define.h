#ifndef PRIMITIV_C_DEFINE_H_
#define PRIMITIV_C_DEFINE_H_

#ifndef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1
#ifndef __cplusplus

#ifndef _Bool
#define _Bool unsigned char
#endif

typedef _Bool bool;
#define true  1
#define false 0

#endif /* __cplusplus */
#endif /* __bool_true_false_are_defined */

#include <stddef.h>
#include <stdint.h>

#endif  // PRIMITIV_C_DEFINE_H_
