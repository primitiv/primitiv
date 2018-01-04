/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_DEFINE_H_
#define PRIMITIV_C_DEFINE_H_

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define PRIMITIV_C_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#ifdef PRIMITIV_C_DLLEXPORT
#define PRIMITIV_C_API __declspec(dllexport)
#else
#define PRIMITIV_C_API __declspec(dllimport)
#endif  // PRIMITIV_C_DLLEXPORT
#else
#define PRIMITIV_C_API
#endif  // __GNUC__, __clang__, _MSC_VER

/*
 * Boolean type.
 */
typedef uint32_t PRIMITIV_C_BOOL;

/*
 * Boolean values.
 * `PRIMITIV_C_TRUE` can not be compared with any `PRIMITIV_C_BOOL` values.
 * Only substituting `PRIMITIV_C_TRUE` to `PRIMITIV_C_BOOL` variables is
 * allowed.
 */
#define PRIMITIV_C_FALSE 0
#define PRIMITIV_C_TRUE 1

#endif  // PRIMITIV_C_DEFINE_H_
