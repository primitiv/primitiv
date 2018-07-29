#include <primitiv/config.h>

#include <primitiv/core/initializer_impl.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/initializer_impl.h>

using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::initializers::Normal;
using primitiv::initializers::Identity;
using primitiv::initializers::XavierUniform;
using primitiv::initializers::XavierNormal;
using primitiv::initializers::XavierUniformConv2D;
using primitiv::initializers::XavierNormalConv2D;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateConstantInitializer(
    float k, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Constant(k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateUniformInitializer(
    float lower, float upper, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Uniform(lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateNormalInitializer(
    float mean, float sd, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Normal(mean, sd));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateIdentityInitializer(
    primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Identity());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateXavierUniformInitializer(
    float scale, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new XavierUniform(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateXavierNormalInitializer(
    float scale, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new XavierNormal(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateXavierUniformConv2DInitializer(
    float scale, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new XavierUniformConv2D(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateXavierNormalConv2DInitializer(
    float scale, primitivInitializer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new XavierNormalConv2D(scale));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
