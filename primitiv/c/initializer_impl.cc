#include <primitiv/config.h>

#include <primitiv/initializer_impl.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/initializer_impl.h>

using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::initializers::Normal;
using primitiv::initializers::Identity;
using primitiv::initializers::XavierUniform;
using primitiv::initializers::XavierNormal;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitiv_initializers_Constant_new(
    float k, primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new Constant(k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_initializers_Uniform_new(
    float lower, float upper, primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new Uniform(lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_initializers_Normal_new(
    float mean, float sd, primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new Normal(mean, sd));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_initializers_Identity_new(
    primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new Identity());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_initializers_XavierUniform_new(
    float scale, primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new XavierUniform(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_initializers_XavierNormal_new(
    float scale, primitivInitializer_t **initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *initializer = to_c_ptr(new XavierNormal(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
