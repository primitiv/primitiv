/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

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

#ifdef SWIG
#define PRIMITIV_CAPI_EXPORT
#else
#if defined(COMPILER_MSVC)
#ifdef PRIMITIV_COMPILE_LIBRARY
#define PRIMITIV_CAPI_EXPORT __declspec(dllexport)
#else
#define PRIMITIV_CAPI_EXPORT __declspec(dllimport)
#endif  // PRIMITIV_COMPILE_LIBRARY
#else
#define PRIMITIV_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // COMPILER_MSVC
#endif  // SWIG
#define CAPI PRIMITIV_CAPI_EXPORT

#endif  // PRIMITIV_C_DEFINE_H_
