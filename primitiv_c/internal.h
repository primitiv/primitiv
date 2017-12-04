#ifndef PRIMITIV_C_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_H_

#include "primitiv_c/define.h"

#include "primitiv/shape.h"
#include "primitiv/tensor.h"

#define CAST_TO_CC_DEVICE(x) reinterpret_cast<Device*>(x)
#define CAST_TO_C_DEVICE(x) reinterpret_cast<primitiv_Device*>(x)
#define CAST_TO_CONST_CC_DEVICE(x) reinterpret_cast<const Device*>(x)
#define CAST_TO_CONST_C_DEVICE(x) reinterpret_cast<const primitiv_Device*>(x)

#define CAST_TO_CC_INITIALIZER(x) reinterpret_cast<Initializer*>(x)
#define CAST_TO_C_INITIALIZER(x) reinterpret_cast<primitiv_Initializer*>(x)
#define CAST_TO_CONST_CC_INITIALIZER(x) reinterpret_cast<const Initializer*>(x)
#define CAST_TO_CONST_C_INITIALIZER(x) reinterpret_cast<const primitiv_Initializer*>(x)

#define CAST_TO_CC_GRAPH(x) reinterpret_cast<Graph*>(x)
#define CAST_TO_C_GRAPH(x) reinterpret_cast<primitiv_Graph*>(x)
#define CAST_TO_CONST_CC_GRAPH(x) reinterpret_cast<const Graph*>(x)
#define CAST_TO_CONST_C_GRAPH(x) reinterpret_cast<const primitiv_Graph*>(x)

#define CAST_TO_CC_MODEL(x) reinterpret_cast<Model*>(x)
#define CAST_TO_C_MODEL(x) reinterpret_cast<primitiv_Model*>(x)
#define CAST_TO_CONST_CC_MODEL(x) reinterpret_cast<const Model*>(x)
#define CAST_TO_CONST_C_MODEL(x) reinterpret_cast<const primitiv_Model*>(x)

#define CAST_TO_CC_PARAMETER(x) reinterpret_cast<Parameter*>(x)
#define CAST_TO_C_PARAMETER(x) reinterpret_cast<primitiv_Parameter*>(x)
#define CAST_TO_CONST_CC_PARAMETER(x) reinterpret_cast<const Parameter*>(x)
#define CAST_TO_CONST_C_PARAMETER(x) reinterpret_cast<const primitiv_Parameter*>(x)

#define CAST_TO_CC_OPTIMIZER(x) reinterpret_cast<Optimizer*>(x)
#define CAST_TO_C_OPTIMIZER(x) reinterpret_cast<primitiv_Optimizer*>(x)
#define CAST_TO_CONST_CC_OPTIMIZER(x) reinterpret_cast<const Optimizer*>(x)
#define CAST_TO_CONST_C_OPTIMIZER(x) reinterpret_cast<const primitiv_Optimizer*>(x)

struct primitiv_Device;

struct primitiv_Initializer;

struct primitiv_Graph;

struct primitiv_Model;

struct primitiv_Parameter;

struct primitiv_Shape {
  primitiv::Shape shape;
};

struct primitiv_Tensor {
  primitiv::Tensor tensor;
};

struct primitiv_Optimizer;

#endif // PRIMITIV_C_INTERNAL_H_
