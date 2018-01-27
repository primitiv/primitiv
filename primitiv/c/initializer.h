#ifndef PRIMITIV_C_INITIALIZER_H_
#define PRIMITIV_C_INITIALIZER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/tensor.h>

/**
 * Opaque type of Initializer.
 */
typedef struct primitivInitializer primitivInitializer_t;

/**
 * Deletes the Initializer object.
 * @param initializer Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Initializer_delete(
    primitivInitializer_t *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_Initializer_apply(
    const primitivInitializer_t *initializer, primitivTensor_t *x);

#endif  // PRIMITIV_C_INITIALIZER_H_
