#include "primitiv_c/internal.h"
#include "primitiv_c/graph.h"

#include <string>
#include <utility>

#include <primitiv/graph.h>

using primitiv::Node;
using primitiv::Graph;

extern "C" {

primitiv_Node primitiv_Node_construct() {
  return {};
}

primitiv_Node primitiv_Node_construct_with_movement(primitiv_Node *node) {
  Node n = std::move(*CAST_TO_CC_NODE(node));
  return *CAST_TO_C_NODE(&n);
}

void primitiv_Node_delete(primitiv_Node *node) {
  delete CAST_TO_CC_GRAPH(node);
}

bool primitiv_Node_valid(const primitiv_Node *node) {
  return CAST_TO_CONST_CC_NODE(node)->valid();
}

primitiv_Graph *primitiv_Node_graph(const primitiv_Node *node) {
  return CAST_TO_C_GRAPH(&CAST_TO_CONST_CC_NODE(node)->graph());
}

uint32_t primitiv_Node_function_id(const primitiv_Node *node) {
  return CAST_TO_CONST_CC_NODE(node)->function_id();
}

uint32_t primitiv_Node_value_id(const primitiv_Node *node) {
  return CAST_TO_CONST_CC_NODE(node)->value_id();
}

const primitiv_Shape* primitiv_Node_shape(const primitiv_Node *node) {
  return new primitiv_Shape{CAST_TO_CONST_CC_NODE(node)->shape()};
}

primitiv_Device* primitiv_Node_device(const primitiv_Node *node) {
  return CAST_TO_C_DEVICE(&CAST_TO_CONST_CC_NODE(node)->device());
}

float primitiv_Node_to_float(const primitiv_Node *node) {
  return CAST_TO_CONST_CC_NODE(node)->to_float();
}

float *primitiv_Node_to_array(const primitiv_Node *node) {
  return &(CAST_TO_CONST_CC_NODE(node)->to_vector())[0];
}

uint32_t *primitiv_Node_argmax(const primitiv_Node *node, uint32_t dim) {
  return &(CAST_TO_CONST_CC_NODE(node)->argmax(dim))[0];
}

uint32_t *primitiv_Node_argmin(const primitiv_Node *node, uint32_t dim) {
  return &(CAST_TO_CONST_CC_NODE(node)->argmin(dim))[0];
}

void primitiv_Node_backward(const primitiv_Node *node) {
  CAST_TO_CONST_CC_NODE(node)->backward();
}

primitiv_Graph *primitiv_Graph_new() {
  return CAST_TO_C_GRAPH(new Graph());
}

void primitiv_Graph_delete(primitiv_Graph *graph) {
  delete CAST_TO_CC_GRAPH(graph);
}

primitiv_Graph *primitiv_Graph_get_default() {
  return CAST_TO_C_GRAPH(&Graph::get_default());
}

void primitiv_Graph_set_default(primitiv_Graph *graph) {
  Graph::set_default(*CAST_TO_CC_GRAPH(graph));
}

void primitiv_Graph_clear(primitiv_Graph *graph) {
  CAST_TO_CC_GRAPH(graph)->clear();
}

const primitiv_Tensor *primitiv_Graph_forward(primitiv_Graph *graph, const primitiv_Node *node) {
  return new primitiv_Tensor{CAST_TO_CC_GRAPH(graph)->forward(*CAST_TO_CONST_CC_NODE(node))};
}

void primitiv_Graph_backward(primitiv_Graph *graph, const primitiv_Node *node) {
  CAST_TO_CC_GRAPH(graph)->backward(*CAST_TO_CONST_CC_NODE(node));
}

const primitiv_Shape *primitiv_Graph_get_shape(const primitiv_Graph *graph, const primitiv_Node *node) {
  return new primitiv_Shape{CAST_TO_CONST_CC_GRAPH(graph)->get_shape(*CAST_TO_CONST_CC_NODE(node))};
}

primitiv_Device *primitiv_Graph_get_device(const primitiv_Graph *graph, const primitiv_Node *node) {
  return CAST_TO_C_DEVICE(&(CAST_TO_CONST_CC_GRAPH(graph)->get_device(*CAST_TO_CONST_CC_NODE(node))));
}

char *primitiv_Graph_dump(const primitiv_Graph *graph, const char *format) {
  std::string str = CAST_TO_CONST_CC_GRAPH(graph)->dump(format);
  unsigned long len = str.length();
  char *c = new char[len + 1];
  std::strncpy(c, str.c_str(), len);
  return c;
}

uint32_t primitiv_Graph_num_functions(const primitiv_Graph *graph) {
  return CAST_TO_CONST_CC_GRAPH(graph)->num_functions();
}

}  // end extern "C"
