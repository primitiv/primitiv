#ifndef PRIMITIV_C_INITIALIZER_H_
#define PRIMITIV_C_INITIALIZER_H_

#include <primitiv/c/define.h>

/**
 * Opaque type of Initializer.
 */
typedef struct primitiv_Initializer primitiv_Initializer;

/**
 * Deletes the Initializer object.
 * @param initializer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Initializer_delete(
    primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Initializer_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

#endif  // PRIMITIV_C_INITIALIZER_H_
