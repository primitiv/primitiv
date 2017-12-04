#ifndef PRIMITIV_C_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_H_

#include "primitiv_c/define.h"
#include "primitiv_c/graph.h"

#include <primitiv/device.h>
#include <primitiv/graph.h>
#include <primitiv/initializer.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <primitiv/optimizer.h>

#define DEFINE_CONVERSION_AS_CAST(cc_name, c_name) \
inline c_name *to_c(primitiv::cc_name *instance) { \
  return reinterpret_cast<c_name*>(instance); \
} \
inline const c_name *to_c(const primitiv::cc_name *instance) { \
  return reinterpret_cast<const c_name*>(instance); \
} \
inline primitiv::cc_name *to_cc(c_name *instance) { \
  return reinterpret_cast<primitiv::cc_name*>(instance); \
} \
inline const primitiv::cc_name *to_cc(const c_name *instance) { \
  return reinterpret_cast<const primitiv::cc_name*>(instance); \
}

struct primitiv_Device;

DEFINE_CONVERSION_AS_CAST(Device, primitiv_Device);

// Definition of Node is in "primitiv_c/graph.h"

DEFINE_CONVERSION_AS_CAST(Node, primitiv_Node);

struct primitiv_Graph;

DEFINE_CONVERSION_AS_CAST(Graph, primitiv_Graph);

struct primitiv_Initializer;

DEFINE_CONVERSION_AS_CAST(Initializer, primitiv_Initializer);

struct primitiv_Model;

DEFINE_CONVERSION_AS_CAST(Model, primitiv_Model);

struct primitiv_Parameter;

DEFINE_CONVERSION_AS_CAST(Parameter, primitiv_Parameter);

struct primitiv_Shape {
  primitiv::Shape shape;
};

inline primitiv_Shape *to_c(primitiv::Shape *instance) {
  return new primitiv_Shape{*instance};
}
inline const primitiv_Shape *to_c(const primitiv::Shape *instance) {
  return const_cast<const primitiv_Shape*>(new primitiv_Shape{*instance});
}
inline primitiv::Shape *to_cc(primitiv_Shape *instance) {
  return &instance->shape;
}
inline const primitiv::Shape *to_cc(const primitiv_Shape *instance) {
  return const_cast<const primitiv::Shape*>(&instance->shape);
}

struct primitiv_Tensor {
  primitiv::Tensor tensor;
};

inline primitiv_Tensor *to_c(primitiv::Tensor *instance) {
  return new primitiv_Tensor{*instance};
}
inline const primitiv_Tensor *to_c(const primitiv::Tensor *instance) {
  return const_cast<const primitiv_Tensor*>(new primitiv_Tensor{*instance});
}
inline primitiv::Tensor *to_cc(primitiv_Tensor *instance) {
  return &instance->tensor;
}
inline const primitiv::Tensor *to_cc(const primitiv_Tensor *instance) {
  return const_cast<const primitiv::Tensor*>(&instance->tensor);
}

struct primitiv_Optimizer;

DEFINE_CONVERSION_AS_CAST(Optimizer, primitiv_Optimizer);

#endif  // PRIMITIV_C_INTERNAL_H_
