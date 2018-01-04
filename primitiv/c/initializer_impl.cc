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
using primitiv::c::internal::to_c_ptr;

primitiv_Status primitiv_initializers_Constant_new(
    float k, primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new Constant(k));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_initializers_Uniform_new(
    float lower, float upper, primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new Uniform(lower, upper));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_initializers_Normal_new(
    float mean, float sd, primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new Normal(mean, sd));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_initializers_Identity_new(
    primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new Identity());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_initializers_XavierUniform_new(
    float scale, primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new XavierUniform(scale));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_initializers_XavierNormal_new(
    float scale, primitiv_Initializer **initializer) try {
  *initializer = to_c_ptr(new XavierNormal(scale));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
