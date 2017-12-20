/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
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

#define CAST_TO_CC_CONSTANT(x) reinterpret_cast<Constant*>(x)
#define CAST_TO_CC_UNIFORM(x) reinterpret_cast<Uniform*>(x)
#define CAST_TO_CC_NORMAL(x) reinterpret_cast<Normal*>(x)
#define CAST_TO_CC_IDENTITY(x) reinterpret_cast<Identity*>(x)
#define CAST_TO_CC_XAVIER_UNIFORM(x) reinterpret_cast<XavierUniform*>(x)
#define CAST_TO_CC_XAVIER_NORMAL(x) reinterpret_cast<XavierNormal*>(x)

extern "C" {

primitiv_Initializer *primitiv_Constant_new(float k) {
  return to_c(new Constant(k));
}
primitiv_Initializer *safe_primitiv_Constant_new(float k,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_Constant_new(k), status, nullptr);
}

void primitiv_Constant_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_CONSTANT(initializer);
}
void safe_primitiv_Constant_delete(primitiv_Initializer *initializer,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_Constant_delete(initializer), status);
}

void primitiv_Constant_apply(const primitiv_Initializer *initializer,
                             primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_Constant_apply(const primitiv_Initializer *initializer,
                                  primitiv_Tensor *x,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Constant_apply(initializer, x), status);
}

primitiv_Initializer *primitiv_Uniform_new(float lower, float upper) {
  return to_c(new Uniform(lower, upper));
}
primitiv_Initializer *safe_primitiv_Uniform_new(float lower,
                                                float upper,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_Uniform_new(lower, upper), status, nullptr);
}

void primitiv_Uniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_UNIFORM(initializer);
}
void safe_primitiv_Uniform_delete(primitiv_Initializer *initializer,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Uniform_delete(initializer), status);
}

void primitiv_Uniform_apply(const primitiv_Initializer *initializer,
                            primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_Uniform_apply(const primitiv_Initializer *initializer,
                                 primitiv_Tensor *x,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_Uniform_apply(initializer, x), status);
}

primitiv_Initializer *primitiv_Normal_new(float mean, float sd) {
  return to_c(new Normal(mean, sd));
}
primitiv_Initializer *safe_primitiv_Normal_new(float mean,
                                               float sd,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Normal_new(mean, sd), status, nullptr);
}

void primitiv_Normal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_NORMAL(initializer);
}
void safe_primitiv_Normal_delete(primitiv_Initializer *initializer,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_Normal_delete(initializer), status);
}

void primitiv_Normal_apply(const primitiv_Initializer *initializer,
                           primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_Normal_apply(const primitiv_Initializer *initializer,
                                primitiv_Tensor *x,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Normal_apply(initializer, x), status);
}

primitiv_Initializer *primitiv_Identity_new() {
  return to_c(new Identity());
}
primitiv_Initializer *safe_primitiv_Identity_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Identity_new(), status, nullptr);
}

void primitiv_Identity_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_IDENTITY(initializer);
}
void safe_primitiv_Identity_delete(primitiv_Initializer *initializer,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_Identity_delete(initializer), status);
}

void primitiv_Identity_apply(const primitiv_Initializer *initializer,
                             primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_Identity_apply(const primitiv_Initializer *initializer,
                                  primitiv_Tensor *x,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Identity_apply(initializer, x), status);
}

primitiv_Initializer *primitiv_XavierUniform_new(float scale) {
  return to_c(new XavierUniform(scale));
}
primitiv_Initializer *safe_primitiv_XavierUniform_new(float scale,
                                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_XavierUniform_new(scale), status, nullptr);
}

void primitiv_XavierUniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_UNIFORM(initializer);
}
void safe_primitiv_XavierUniform_delete(primitiv_Initializer *initializer,
                                        primitiv_Status *status) {
  SAFE_EXPR(primitiv_XavierUniform_delete(initializer), status);
}

void primitiv_XavierUniform_apply(const primitiv_Initializer *initializer,
                                  primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_XavierUniform_apply(const primitiv_Initializer *initializer,
                                       primitiv_Tensor *x,
                                       primitiv_Status *status) {
  SAFE_EXPR(primitiv_XavierUniform_apply(initializer, x), status);
}

primitiv_Initializer *primitiv_XavierNormal_new(float scale) {
  return to_c(new XavierNormal(scale));
}
primitiv_Initializer *safe_primitiv_XavierNormal_new(float scale,
                                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_XavierNormal_new(scale), status, nullptr);
}

void primitiv_XavierNormal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_NORMAL(initializer);
}
void safe_primitiv_XavierNormal_delete(primitiv_Initializer *initializer,
                                       primitiv_Status *status) {
  SAFE_EXPR(primitiv_XavierNormal_delete(initializer), status);
}

void primitiv_XavierNormal_apply(const primitiv_Initializer *initializer,
                                 primitiv_Tensor *x) {
  to_cc(initializer)->apply(*to_cc(x));
}
void safe_primitiv_XavierNormal_apply(const primitiv_Initializer *initializer,
                                      primitiv_Tensor *x,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_XavierNormal_apply(initializer, x), status);
}

}  // end extern "C"
