#ifndef PRIMITIV_C_DEFINE_H_
#define PRIMITIV_C_DEFINE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define PRIMITIV_C_EXTERN extern "C"
#else
#define PRIMITIV_C_EXTERN extern
#endif  // __cplusplus

#if defined(__GNUC__) || defined(__clang__)
#define PRIMITIV_C_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#ifdef PRIMITIV_C_DLLEXPORT
#define PRIMITIV_C_EXPORT __declspec(dllexport)
#else
#define PRIMITIV_C_EXPORT __declspec(dllimport)
#endif  // PRIMITIV_C_DLLEXPORT
#else
#define PRIMITIV_C_EXPORT
#endif  // __GNUC__, __clang__, _MSC_VER

#define PRIMITIV_C_API PRIMITIV_C_EXTERN PRIMITIV_C_EXPORT

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
#define PRIMITIV_C_FALSE 0u
#define PRIMITIV_C_TRUE 1u

/*
 * Return codes.
 */
typedef int32_t PRIMITIV_C_STATUS;
#define PRIMITIV_C_OK 0
#define PRIMITIV_C_ERROR -1

#endif  // PRIMITIV_C_DEFINE_H_
