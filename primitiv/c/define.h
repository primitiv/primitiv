/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_DEFINE_H_
#define PRIMITIV_C_DEFINE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef __bool_true_false_are_defined
#error "primitiv C API requires boolean support (C99/C++)."
#endif  // __bool_true_false_are_defined

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

#endif  // PRIMITIV_C_DEFINE_H_
