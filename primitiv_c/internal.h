#ifndef PRIMITIV_C_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_H_

#include "primitiv_c/api.h"

#include "primitiv/primitiv.h"

struct primitiv_Device {
  primitiv::Device device;
};

struct primitiv_Graph {
  primitiv::Graph graph;
};

struct primitiv_Initializer {
  primitiv::Initializer initializer;
};

struct primitiv_Model {
  primitiv::Model model;
};

struct primitiv_Node {
  primitiv::Node node;
};

struct primitiv_Parameter {
  primitiv::Parameter parameter;
};

struct primitiv_Shape {
  primitiv::Shape shape;
};

struct primitiv_Tensor {
  primitiv::Tensor tensor;
};

struct primitiv_Optimizer {
  primitiv::Optimizer optimizer;
};

#endif //PRIMITIV_C_INTERNAL_H_
