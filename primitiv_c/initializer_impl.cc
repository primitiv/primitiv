#include "primitiv_c/internal.h"
#include "primitiv_c/initializer_impl.h"

#include <primitiv/initializer_impl.h>

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

void primitiv_Constant_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_CONSTANT(initializer);
}

primitiv_Initializer *primitiv_Uniform_new(float lower, float upper) {
  return to_c(new Uniform(lower, upper));
}

void primitiv_Uniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_UNIFORM(initializer);
}

primitiv_Initializer *primitiv_Normal_new(float mean, float sd) {
  return to_c(new Normal(mean, sd));
}

void primitiv_Normal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_NORMAL(initializer);
}

primitiv_Initializer *primitiv_Identity_new() {
  return to_c(new Identity());
}

void primitiv_Identity_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_IDENTITY(initializer);
}

primitiv_Initializer *primitiv_XavierUniform_new(float scale) {
  return to_c(new XavierUniform(scale));
}

void primitiv_XavierUniform_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_UNIFORM(initializer);

}

primitiv_Initializer *primitiv_XavierNormal_new(float scale) {
  return to_c(new XavierNormal(scale));
}

void primitiv_XavierNormal_delete(primitiv_Initializer *initializer) {
  delete CAST_TO_CC_XAVIER_NORMAL(initializer);
}

}  // end extern "C"
