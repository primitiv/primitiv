/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/initializer_impl.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/initializer_impl.h>

using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::initializers::Normal;
using primitiv::initializers::Identity;
using primitiv::initializers::XavierUniform;
using primitiv::initializers::XavierNormal;
using primitiv::c::internal::to_c;
using primitiv::c::internal::to_cc;

#define CAST_TO_CC_CONSTANT(x) reinterpret_cast<Constant*>(x)
#define CAST_TO_CONST_CC_CONSTANT(x) reinterpret_cast<const Constant*>(x)
#define CAST_TO_CC_UNIFORM(x) reinterpret_cast<Uniform*>(x)
#define CAST_TO_CONST_CC_UNIFORM(x) reinterpret_cast<const Uniform*>(x)
#define CAST_TO_CC_NORMAL(x) reinterpret_cast<Normal*>(x)
#define CAST_TO_CONST_CC_NORMAL(x) reinterpret_cast<const Normal*>(x)
#define CAST_TO_CC_IDENTITY(x) reinterpret_cast<Identity*>(x)
#define CAST_TO_CONST_CC_IDENTITY(x) reinterpret_cast<const Identity*>(x)
#define CAST_TO_CC_XAVIER_UNIFORM(x) reinterpret_cast<XavierUniform*>(x)
#define CAST_TO_CONST_CC_XAVIER_UNIFORM(x) \
reinterpret_cast<const XavierUniform*>(x)
#define CAST_TO_CC_XAVIER_NORMAL(x) reinterpret_cast<XavierNormal*>(x)
#define CAST_TO_CONST_CC_XAVIER_NORMAL(x) \
reinterpret_cast<const XavierNormal*>(x)

extern "C" {

primitiv_Initializer *primitiv_Constant_new(float k) {
  return to_c(new Constant(k));
}

void primitiv_Constant_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_CONSTANT(initializer);
}

primitiv_Status primitiv_Constant_apply(const primitiv_Initializer *initializer,
                                        primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_CONSTANT(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Initializer *primitiv_Uniform_new(float lower, float upper) {
  return to_c(new Uniform(lower, upper));
}

void primitiv_Uniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_UNIFORM(initializer);
}

primitiv_Status primitiv_Uniform_apply(const primitiv_Initializer *initializer,
                                       primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_UNIFORM(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Initializer *primitiv_Normal_new(float mean, float sd) {
  return to_c(new Normal(mean, sd));
}

void primitiv_Normal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_NORMAL(initializer);
}

primitiv_Status primitiv_Normal_apply(const primitiv_Initializer *initializer,
                           primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_NORMAL(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Initializer *primitiv_Identity_new() {
  return to_c(new Identity());
}

void primitiv_Identity_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_IDENTITY(initializer);
}

primitiv_Status primitiv_Identity_apply(const primitiv_Initializer *initializer,
                             primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_IDENTITY(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Initializer *primitiv_XavierUniform_new(float scale) {
  return to_c(new XavierUniform(scale));
}

void primitiv_XavierUniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_UNIFORM(initializer);
}

primitiv_Status primitiv_XavierUniform_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_XAVIER_UNIFORM(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Initializer *primitiv_XavierNormal_new(float scale) {
  return to_c(new XavierNormal(scale));
}

void primitiv_XavierNormal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_NORMAL(initializer);
}

primitiv_Status primitiv_XavierNormal_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x) {
  try {
    CAST_TO_CONST_CC_XAVIER_NORMAL(initializer)->apply(*to_cc(x));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"

#undef CAST_TO_CC_CONSTANT
#undef CAST_TO_CONST_CC_CONSTANT
#undef CAST_TO_CC_UNIFORM
#undef CAST_TO_CONST_CC_UNIFORM
#undef CAST_TO_CC_NORMAL
#undef CAST_TO_CONST_CC_NORMAL
#undef CAST_TO_CC_IDENTITY
#undef CAST_TO_CONST_CC_IDENTITY
#undef CAST_TO_CC_XAVIER_UNIFORM
#undef CAST_TO_CONST_CC_XAVIER_UNIFORM
#undef CAST_TO_CC_XAVIER_NORMAL
#undef CAST_TO_CONST_CC_XAVIER_NORMAL
