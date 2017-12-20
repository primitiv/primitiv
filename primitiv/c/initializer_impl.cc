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
using primitiv::c::internal::to_c_ptr;

extern "C" {

primitiv_Initializer *primitiv_initializers_Constant_new(float k) {
  return to_c_ptr(new Constant(k));
}

primitiv_Initializer *primitiv_initializers_Uniform_new(float lower,
                                                        float upper) {
  return to_c_ptr(new Uniform(lower, upper));
}

primitiv_Initializer *primitiv_initializers_Normal_new(float mean, float sd) {
  return to_c_ptr(new Normal(mean, sd));
}

primitiv_Initializer *primitiv_initializers_Identity_new() {
  return to_c_ptr(new Identity());
}

primitiv_Initializer *primitiv_initializers_XavierUniform_new(float scale) {
  return to_c_ptr(new XavierUniform(scale));
}

primitiv_Initializer *primitiv_initializers_XavierNormal_new(float scale) {
  return to_c_ptr(new XavierNormal(scale));
}

}  // end extern "C"
