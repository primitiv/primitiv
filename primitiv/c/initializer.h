/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_INITIALIZER_H_
#define PRIMITIV_C_INITIALIZER_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque type of Initializer.
 */
typedef struct primitiv_Initializer primitiv_Initializer;

/**
 * Deletes the Initializer object.
 * @param initializer Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Initializer_delete(primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Initializer_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_H_
